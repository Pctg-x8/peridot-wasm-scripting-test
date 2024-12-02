use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::{Arc, RwLock},
};

use bedrock::{
    self as br, CommandBufferMut, CommandPoolMut, DescriptorPoolMut, Device,
    GraphicsPipelineBuilder, Image, ImageChild, ImageSubresourceSlice, RenderPass, SubmissionBatch,
};
use peridot::NativeLinker;
use peridot_command_object::{
    BeginRenderPass, BindGraphicsDescriptorSets, BindGraphicsPipeline, EndRenderPass,
    GraphicsCommand, GraphicsCommandCombiner, RangedBuffer, StandardIndexedMesh,
};
use peridot_math::{One, Zero};
use peridot_memory_manager::{BufferMapMode, MemoryManager};
use peridot_vertex_processing_pack::PvpShaderModules;
use uuid::Uuid;
use wasmtime::{
    component::{ComponentType, Lift, Lower, Resource, ResourceTable, ResourceType},
    StoreContextMut,
};

#[repr(C)]
#[derive(Clone)]
pub struct CameraUniformData {
    pub view_projection_matrix: peridot_math::Matrix4F32,
}

#[repr(C)]
#[derive(Clone)]
pub struct ObjectUniformData {
    pub object_matrix: peridot_math::Matrix4F32,
}

#[repr(C)]
#[derive(Clone)]
pub struct Vertex {
    pos: peridot_math::Vector4F32,
    normal: peridot_math::Vector4F32,
}

pub struct CubeTransformComponent {
    position: peridot_math::Vector3F32,
    rotation: peridot_math::QuaternionF32,
    scale: peridot_math::Vector3F32,
}

pub struct Renderer {
    render_target_size: peridot::math::Vector2<u32>,
    main_render_pass: br::RenderPassObject<peridot::DeviceObject>,
    depth_buffer: Arc<br::ImageViewObject<peridot_memory_manager::Image>>,
    device_buffer: peridot_memory_manager::Buffer,
    cube_vertex_offset: br::vk::VkDeviceSize,
    cube_index_offset: br::vk::VkDeviceSize,
    camera: peridot_math::Camera,
    camera_uniform_offset: br::vk::VkDeviceSize,
    object_uniform_offset: br::vk::VkDeviceSize,
    object_upload_buffer: peridot_memory_manager::Buffer,
    _descriptor_pool: br::DescriptorPoolObject<peridot::DeviceObject>,
    camera_descriptor_set: br::DescriptorSet,
    object_descriptor_set: br::DescriptorSet,
    _dsl_ub1: br::DescriptorSetLayoutObject<peridot::DeviceObject>,
    cube_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    cube_pipeline_layout: br::PipelineLayoutObject<peridot::DeviceObject>,
    cube_pipeline: br::PipelineObject<peridot::DeviceObject>,
    cube_transform_component: Arc<RwLock<CubeTransformComponent>>,
}
impl Renderer {
    fn new(
        e: &mut peridot::Engine<impl NativeLinker>,
        memory_manager: &mut MemoryManager,
    ) -> br::Result<Self> {
        let frame_size = e.back_buffer(0).unwrap().image().size().clone();
        let rect = frame_size
            .as_2d_ref()
            .clone()
            .into_rect(br::vk::VkOffset2D::ZERO);
        let viewport = rect.make_viewport(0.0..1.0);

        let (bb_final_layout, bb_final_transition_at) = e.requesting_back_buffer_layout();
        let main_render_pass = br::RenderPassBuilder2::new(
            &[
                br::AttachmentDescription2::new(e.back_buffer_format())
                    .with_layout_from(br::ImageLayout::Undefined.to(bb_final_layout))
                    .color_memory_op(br::LoadOp::Clear, br::StoreOp::Store),
                br::AttachmentDescription2::new(br::vk::VK_FORMAT_D24_UNORM_S8_UINT)
                    .with_layout_from(
                        br::ImageLayout::Undefined.to(br::ImageLayout::DepthStencilAttachmentOpt),
                    )
                    .color_memory_op(br::LoadOp::Clear, br::StoreOp::DontCare)
                    .stencil_memory_op(br::LoadOp::Clear, br::StoreOp::DontCare),
            ],
            &[br::SubpassDescription2::new()
                .colors(&[br::AttachmentReference2::color(
                    0,
                    br::ImageLayout::ColorAttachmentOpt,
                )])
                .depth_stencil(&br::AttachmentReference2::depth_stencil(
                    1,
                    br::ImageLayout::DepthStencilAttachmentOpt,
                ))],
            &[br::SubpassDependency2::new(
                br::SubpassIndex::Internal(0),
                br::SubpassIndex::External,
            )
            .of_memory(
                br::AccessFlags::COLOR_ATTACHMENT.write,
                br::AccessFlags::MEMORY.read,
            )
            .of_execution(
                br::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                bb_final_transition_at,
            )
            .by_region()],
        )
        .create(e.graphics_device().clone())?;

        let depth_buffer = memory_manager
            .allocate_device_local_image(
                e.graphics(),
                br::ImageDesc::new(
                    frame_size.as_2d_ref().clone(),
                    br::vk::VK_FORMAT_D24_UNORM_S8_UINT,
                )
                .as_depth_stencil_attachment(),
            )?
            .subresource_range(br::AspectMask::DEPTH.stencil(), 0..1, 0..1)
            .view_builder()
            .create()?;

        let mut camera = peridot_math::Camera {
            projection: Some(peridot_math::ProjectionMethod::Perspective {
                fov: 60.0f32.to_radians(),
            }),
            position: peridot_math::Vector3(0.0, 2.0, -5.0),
            rotation: peridot_math::Quaternion::ONE,
            depth_range: 0.3..100.0,
        };
        camera.look_at(peridot_math::Vector3::ZERO);

        let (
            device_buffer,
            [camera_uniform_offset, object_uniform_offset, cube_vertex_offset, cube_index_offset],
        ) = memory_manager.allocate_device_local_buffer_with_content_array(
            e.graphics(),
            &[
                peridot::BufferContent::uniform::<CameraUniformData>(),
                peridot::BufferContent::uniform::<ObjectUniformData>(),
                peridot::BufferContent::vertices::<Vertex>(24),
                peridot::BufferContent::indices::<u16>(36),
            ],
            br::BufferUsage::TRANSFER_DEST,
        )?;
        #[repr(C)]
        pub struct BufferInitializationContents {
            camera_uniform: CameraUniformData,
            object_uniform: ObjectUniformData,
            cube_vertices: [Vertex; 24],
            cube_indices: [u16; 36],
        }
        let mut init_buffer = memory_manager.allocate_upload_buffer(
            e.graphics(),
            br::BufferDesc::new_for_type::<BufferInitializationContents>(
                br::BufferUsage::TRANSFER_SRC,
            ),
        )?;
        init_buffer.write_content(BufferInitializationContents {
            camera_uniform: CameraUniformData {
                view_projection_matrix: camera
                    .view_projection_matrix(frame_size.width as f32 / frame_size.height as f32),
            },
            object_uniform: ObjectUniformData {
                object_matrix: peridot_math::Matrix4::ONE,
            },
            cube_vertices: [
                // +X
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(1.0, 0.0, 0.0, 0.0),
                },
                // +Y
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 1.0, 0.0, 0.0),
                },
                // +Z
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, 1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, 1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, 1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, 1.0, 0.0),
                },
                // -X
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(-1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(-1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(-1.0, 0.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(-1.0, 0.0, 0.0, 0.0),
                },
                // -Y
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, -1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, 0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, -1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, -1.0, 0.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, -1.0, 0.0, 0.0),
                },
                // -Z
                Vertex {
                    pos: peridot_math::Vector4(0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, -1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, -1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, 0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, -1.0, 0.0),
                },
                Vertex {
                    pos: peridot_math::Vector4(-0.5, -0.5, -0.5, 1.0),
                    normal: peridot_math::Vector4(0.0, 0.0, -1.0, 0.0),
                },
            ],
            cube_indices: [
                0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7, 8, 9, 10, 10, 9, 11, 12, 13, 14, 14, 13, 15,
                16, 17, 18, 18, 17, 19, 20, 21, 22, 22, 21, 23,
            ],
        })?;

        e.submit_commands(|rec| {
            rec.copy_buffer(
                &init_buffer,
                &device_buffer,
                &[
                    br::BufferCopy::copy_data::<CameraUniformData>(
                        core::mem::offset_of!(BufferInitializationContents, camera_uniform) as _,
                        camera_uniform_offset,
                    ),
                    br::BufferCopy::copy_data::<ObjectUniformData>(
                        core::mem::offset_of!(BufferInitializationContents, object_uniform) as _,
                        object_uniform_offset,
                    ),
                    br::BufferCopy::copy_data::<[Vertex; 24]>(
                        core::mem::offset_of!(BufferInitializationContents, cube_vertices) as _,
                        cube_vertex_offset,
                    ),
                    br::BufferCopy::copy_data::<[u16; 36]>(
                        core::mem::offset_of!(BufferInitializationContents, cube_indices) as _,
                        cube_index_offset,
                    ),
                ],
            )
            .pipeline_barrier_2(&br::DependencyInfo::new(
                &[br::MemoryBarrier2::new()
                    .from(
                        br::PipelineStageFlags2::COPY,
                        br::AccessFlags2::TRANSFER.write,
                    )
                    .to(
                        br::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                            | br::PipelineStageFlags2::INDEX_INPUT
                            | br::PipelineStageFlags2::VERTEX_SHADER,
                        br::AccessFlags2::VERTEX_ATTRIBUTE_READ
                            | br::AccessFlags2::INDEX_READ
                            | br::AccessFlags2::UNIFORM_READ,
                    )],
                &[],
                &[],
            ))
        })?;

        let object_upload_buffer = memory_manager.allocate_upload_buffer(
            e.graphics(),
            br::BufferDesc::new_for_type::<ObjectUniformData>(br::BufferUsage::TRANSFER_SRC),
        )?;

        let dsl_ub1 = br::DescriptorSetLayoutBuilder::new(&[br::DescriptorType::UniformBuffer
            .make_binding(0, 1)
            .only_for_vertex()])
        .create(e.graphics_device().clone())?;

        let shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.plain").expect("Failed to load shader"),
        )?;
        let cube_pl = br::PipelineLayoutBuilder::new(
            &[
                br::DescriptorSetLayoutObjectRef::new(&dsl_ub1),
                br::DescriptorSetLayoutObjectRef::new(&dsl_ub1),
            ],
            &[],
        )
        .create(e.graphics_device().clone())?;
        let mut cube_pipeline = br::NonDerivedGraphicsPipelineBuilder::new(
            &cube_pl,
            main_render_pass.subpass(0),
            shader.generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
        );
        cube_pipeline
            .viewport_scissors(
                br::DynamicArrayState::Static(&[viewport]),
                br::DynamicArrayState::Static(&[rect]),
            )
            .add_attachment_blend(br::AttachmentColorBlendState::premultiplied())
            .depth_test_settings(Some(br::CompareOp::LessOrEqual), true)
            .multisample_state(Some(br::MultisampleState::new()));
        let cube_pipeline = cube_pipeline.create(
            e.graphics_device().clone(),
            None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
        )?;

        let mut dp =
            br::DescriptorPoolBuilder::new(2, &[br::DescriptorType::UniformBuffer.make_size(2)])
                .create(e.graphics_device().clone())?;
        let [camera_descriptor_set, object_descriptor_set] = dp.alloc_array(&[
            br::DescriptorSetLayoutObjectRef::new(&dsl_ub1),
            br::DescriptorSetLayoutObjectRef::new(&dsl_ub1),
        ])?;
        e.graphics_device().update_descriptor_sets(
            &[
                camera_descriptor_set
                    .binding_at(0)
                    .write(br::DescriptorContents::uniform_buffer(
                        &device_buffer,
                        camera_uniform_offset
                            ..camera_uniform_offset
                                + (core::mem::size_of::<CameraUniformData>() as u64),
                    )),
                object_descriptor_set
                    .binding_at(0)
                    .write(br::DescriptorContents::uniform_buffer(
                        &device_buffer,
                        object_uniform_offset
                            ..object_uniform_offset
                                + (core::mem::size_of::<ObjectUniformData>() as u64),
                    )),
            ],
            &[],
        );

        Ok(Self {
            render_target_size: peridot::math::Vector2(frame_size.width, frame_size.height),
            main_render_pass,
            depth_buffer: Arc::new(depth_buffer),
            device_buffer,
            cube_vertex_offset,
            cube_index_offset,
            camera,
            camera_uniform_offset,
            object_uniform_offset,
            object_upload_buffer,
            _descriptor_pool: dp,
            camera_descriptor_set,
            object_descriptor_set,
            _dsl_ub1: dsl_ub1,
            cube_shader: shader,
            cube_pipeline_layout: cube_pl,
            cube_pipeline,
            cube_transform_component: Arc::new(RwLock::new(CubeTransformComponent {
                position: peridot_math::Vector3::ZERO,
                rotation: peridot_math::Quaternion::ONE,
                scale: peridot_math::Vector3::ONE,
            })),
        })
    }

    fn resize(
        &mut self,
        e: &mut peridot::Engine<impl NativeLinker>,
        memory_manager: &mut MemoryManager,
        new_size: peridot_math::Vector2<u32>,
    ) -> br::Result<()> {
        self.render_target_size = new_size;

        let rect = br::vk::VkExtent2D::from(new_size).into_rect(br::vk::VkOffset2D::ZERO);
        let viewport = rect.make_viewport(0.0..1.0);

        self.depth_buffer = Arc::new(
            memory_manager
                .allocate_device_local_image(
                    e.graphics(),
                    br::ImageDesc::new(new_size, br::vk::VK_FORMAT_D24_UNORM_S8_UINT)
                        .as_depth_stencil_attachment(),
                )?
                .subresource_range(br::AspectMask::DEPTH.stencil(), 0..1, 0..1)
                .view_builder()
                .create()?,
        );

        let mut cube_pipeline = br::NonDerivedGraphicsPipelineBuilder::new(
            &self.cube_pipeline_layout,
            self.main_render_pass.subpass(0),
            self.cube_shader
                .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
        );
        cube_pipeline
            .viewport_scissors(
                br::DynamicArrayState::Static(&[viewport]),
                br::DynamicArrayState::Static(&[rect]),
            )
            .add_attachment_blend(br::AttachmentColorBlendState::premultiplied())
            .depth_test_settings(Some(br::CompareOp::LessOrEqual), true)
            .multisample_state(Some(br::MultisampleState::new()));
        self.cube_pipeline = cube_pipeline.create(
            e.graphics_device().clone(),
            None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
        )?;

        Ok(())
    }

    fn populate_main_commands(
        &self,
        cb: &mut br::SynchronizedCommandBuffer<
            impl br::CommandPoolMut<ConcreteDevice = peridot::DeviceObject>,
            impl br::CommandBufferMut,
        >,
        fb: &br::FramebufferObject<peridot::DeviceObject>,
    ) -> br::Result<()> {
        let cube_mesh = StandardIndexedMesh {
            vertex_buffers: vec![RangedBuffer::for_type::<[Vertex; 24]>(
                &self.device_buffer,
                self.cube_vertex_offset,
            )],
            index_buffer: RangedBuffer::for_type::<[u16; 36]>(
                &self.device_buffer,
                self.cube_index_offset,
            ),
            index_type: br::IndexType::U16,
            vertex_count: 36,
        };

        cube_mesh
            .draw(1)
            .after_of((
                BindGraphicsPipeline(&self.cube_pipeline),
                BindGraphicsDescriptorSets::new(
                    &self.cube_pipeline_layout,
                    &[self.camera_descriptor_set, self.object_descriptor_set],
                ),
            ))
            .between(
                BeginRenderPass::new(
                    &self.main_render_pass,
                    fb,
                    br::vk::VkRect2D {
                        offset: br::vk::VkOffset2D::ZERO,
                        extent: self.render_target_size.into(),
                    },
                )
                .with_clear_values(vec![
                    br::ClearValue::color_f32([0.0, 0.0, 0.2, 1.0]),
                    br::ClearValue::depth_stencil(1.0, 0),
                ]),
                EndRenderPass,
            )
            .execute_and_finish(cb.begin()?.as_dyn_ref())
    }
}

pub struct ScriptComponentInstance {
    store: wasmtime::Store<ScriptComponentInstanceState>,
    instance: wasmtime::component::Instance,
    on_message_func: wasmtime::component::TypedFunc<(u32,), ()>,
}
impl ScriptComponentInstance {
    pub fn new(
        mut store: wasmtime::Store<ScriptComponentInstanceState>,
        instance: wasmtime::component::Instance,
    ) -> Self {
        let on_message_func = instance.get_typed_func(&mut store, "on-message").unwrap();

        Self {
            store,
            instance,
            on_message_func,
        }
    }

    pub fn dispatch_message(&mut self, msg_id: u32) -> wasmtime::Result<()> {
        self.on_message_func.call(&mut self.store, (msg_id,))?;
        self.on_message_func.post_return(&mut self.store)?;

        Ok(())
    }
}

pub struct ScriptComponentInstanceState {
    id: Uuid,
    update_subscriptions_ref: Arc<RwLock<HashSet<(Uuid, u32)>>>,
    delta_time_seconds: Arc<RwLock<f32>>,
    resource_table: ResourceTable,
}

pub struct ScriptingEngine {
    engine: wasmtime::Engine,
    update_subscriptions: Arc<RwLock<HashSet<(Uuid, u32)>>>,
    delta_time_seconds: Arc<RwLock<f32>>,
    instance_map: HashMap<Uuid, ScriptComponentInstance>,
}
impl ScriptingEngine {
    pub fn new() -> Self {
        let engine = wasmtime::Engine::default();

        Self {
            engine,
            update_subscriptions: Arc::new(RwLock::new(HashSet::new())),
            delta_time_seconds: Arc::new(RwLock::new(0.0)),
            instance_map: HashMap::new(),
        }
    }

    pub fn update(&mut self, dt_sec: f32) {
        *self.delta_time_seconds.write().unwrap() = dt_sec;

        for (iid, mid) in self.update_subscriptions.read().unwrap().iter() {
            self.instance_map
                .get_mut(iid)
                .unwrap()
                .dispatch_message(*mid)
                .expect("Failed to dispatch message");
        }
    }
}

pub struct ScriptingEngineTransformComponentRefResource {
    entity_ref: Arc<RwLock<CubeTransformComponent>>,
}
impl ScriptingEngineTransformComponentRefResource {
    pub fn position(&self) -> peridot_math::Vector3F32 {
        self.entity_ref.read().unwrap().position.clone()
    }

    pub fn rotation(&self) -> peridot_math::QuaternionF32 {
        self.entity_ref.read().unwrap().rotation.clone()
    }

    pub fn scale(&self) -> peridot_math::Vector3F32 {
        self.entity_ref.read().unwrap().scale.clone()
    }

    pub fn set_position(&self, pos: peridot_math::Vector3F32) {
        self.entity_ref.write().unwrap().position = pos;
    }

    pub fn set_rotation(&self, rot: peridot_math::QuaternionF32) {
        self.entity_ref.write().unwrap().rotation = rot;
    }

    pub fn set_scale(&self, scale: peridot_math::Vector3F32) {
        self.entity_ref.write().unwrap().scale = scale;
    }
}

#[derive(ComponentType, Lower, Lift)]
#[component(record)]
pub struct ScriptingEngineMathVector3 {
    x: f32,
    y: f32,
    z: f32,
}
impl From<peridot_math::Vector3F32> for ScriptingEngineMathVector3 {
    fn from(value: peridot_math::Vector3F32) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}
impl Into<peridot_math::Vector3F32> for ScriptingEngineMathVector3 {
    fn into(self) -> peridot_math::Vector3F32 {
        peridot_math::Vector3(self.x, self.y, self.z)
    }
}

#[derive(ComponentType, Lower, Lift)]
#[component(record)]
pub struct ScriptingEngineMathQuaternion {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
impl From<peridot_math::QuaternionF32> for ScriptingEngineMathQuaternion {
    fn from(value: peridot_math::QuaternionF32) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
            w: value.3,
        }
    }
}
impl Into<peridot_math::QuaternionF32> for ScriptingEngineMathQuaternion {
    fn into(self) -> peridot_math::QuaternionF32 {
        peridot_math::Quaternion(self.x, self.y, self.z, self.w)
    }
}

pub struct ScriptingEngineEntityResource {
    entity_ref: Arc<RwLock<CubeTransformComponent>>,
}
impl ScriptingEngineEntityResource {
    pub fn transform(&self) -> ScriptingEngineTransformComponentRefResource {
        ScriptingEngineTransformComponentRefResource {
            entity_ref: self.entity_ref.clone(),
        }
    }
}

pub async fn game_main(e: &mut peridot::Engine<impl peridot::NativeLinker>) {
    let mut memory_manager = MemoryManager::new(e.graphics());
    let mut renderer =
        Renderer::new(e, &mut memory_manager).expect("Failed to initialize renderer");

    let mut frame_size = e.back_buffer(0).unwrap().image().size().as_2d_ref().clone();
    let mut backbuffer_resources = e.iter_back_buffers().cloned().collect::<Vec<_>>();
    let mut frame_buffers = backbuffer_resources
        .iter()
        .map(|bb| {
            br::FramebufferBuilder::new(&renderer.main_render_pass)
                .with_attachment(bb)
                .with_attachment(&renderer.depth_buffer)
                .create()
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to create framebuffer");

    let mut main_command_pool = br::CommandPoolBuilder::new(e.graphics_queue_family_index())
        .create(e.graphics_device().clone())
        .expect("Failed to create main command pool");
    let mut main_command_buffers = main_command_pool
        .alloc(e.back_buffer_count() as _, true)
        .expect("Failed to allocate main command buffers");
    for (cb, fb) in main_command_buffers.iter_mut().zip(frame_buffers.iter()) {
        renderer
            .populate_main_commands(
                &mut unsafe { cb.synchronize_with(&mut main_command_pool) },
                fb,
            )
            .expect("Failed to populate main commands");
    }

    let mut update_command_pool = br::CommandPoolBuilder::new(e.graphics_queue_family_index())
        .create(e.graphics_device().clone())
        .expect("Failed to create update command pool");
    let [mut update_command_buffer] = update_command_pool
        .alloc_array::<1>(true)
        .expect("Failed to allocate update command buffer");

    let mut scripting_engine = ScriptingEngine::new();
    let script_path = std::env::current_dir().unwrap().join(
        "../../../peridot-wasm-scripting-test/script/target/wasm32-unknown-unknown/release/script.wasm",
    );
    println!("script path: {}", script_path.display());
    let script_bin = std::fs::read(script_path).expect("Failed to load script");
    let script_component =
        wasmtime::component::Component::new(&scripting_engine.engine, &script_bin)
            .expect("Failed to create script component");
    let mut linker = wasmtime::component::Linker::new(&scripting_engine.engine);
    let mut engine_instance = linker
        .instance("peridot:core/engine")
        .expect("Failed to create external instance");
    engine_instance
        .func_wrap("log", |_store, (text,): (String,)| {
            println!("script log: {text}");
            Ok(())
        })
        .expect("Failed to register global func");
    engine_instance
        .func_wrap(
            "subscribe-update",
            |store: StoreContextMut<ScriptComponentInstanceState>, (id,): (u32,)| {
                let instance_id = store.data().id.clone();
                store
                    .data()
                    .update_subscriptions_ref
                    .write()
                    .unwrap()
                    .insert((instance_id, id));

                Ok(())
            },
        )
        .expect("Failed to register global func");
    engine_instance
        .func_wrap(
            "unsubscribe-update",
            |store: StoreContextMut<ScriptComponentInstanceState>, (id,): (u32,)| {
                let instance_id = store.data().id.clone();
                store
                    .data()
                    .update_subscriptions_ref
                    .write()
                    .unwrap()
                    .remove(&(instance_id, id));

                Ok(())
            },
        )
        .expect("Failed to register global func");
    engine_instance
        .func_wrap("cube-transform", {
            let entity_ref = renderer.cube_transform_component.clone();

            move |mut store: StoreContextMut<ScriptComponentInstanceState>, _params: ()| {
                Ok((store
                    .data_mut()
                    .resource_table
                    .push(ScriptingEngineTransformComponentRefResource {
                        entity_ref: entity_ref.clone(),
                    })
                    .unwrap(),))
            }
        })
        .expect("Failed to register engine export fn");
    engine_instance
        .func_wrap(
            "delta-time-seconds",
            |store: StoreContextMut<ScriptComponentInstanceState>, _params: ()| {
                Ok((*store.data().delta_time_seconds.read().unwrap(),))
            },
        )
        .expect("Failed to register global func");
    engine_instance
        .resource(
            "transform-component",
            ResourceType::host::<ScriptingEngineTransformComponentRefResource>(),
            |_state, _handle| Ok(()),
        )
        .expect("Failed to register transform-component resource");
    engine_instance
        .func_wrap(
            "[method]transform-component.position",
            |store: StoreContextMut<ScriptComponentInstanceState>,
             (this,): (Resource<ScriptingEngineTransformComponentRefResource>,)| {
                Ok((ScriptingEngineMathVector3::from(
                    store.data().resource_table.get(&this).unwrap().position(),
                ),))
            },
        )
        .expect("Failed to set transform-component.position function impl");
    engine_instance
        .func_wrap(
            "[method]transform-component.rotation",
            |store: StoreContextMut<ScriptComponentInstanceState>,
             (this,): (Resource<ScriptingEngineTransformComponentRefResource>,)| {
                Ok((ScriptingEngineMathQuaternion::from(
                    store.data().resource_table.get(&this).unwrap().rotation(),
                ),))
            },
        )
        .expect("Failed to set transform-component.position function impl");
    engine_instance
        .func_wrap(
            "[method]transform-component.set-rotation",
            |store: StoreContextMut<ScriptComponentInstanceState>,
             (this, rot): (
                Resource<ScriptingEngineTransformComponentRefResource>,
                ScriptingEngineMathQuaternion,
            )| {
                store
                    .data()
                    .resource_table
                    .get(&this)
                    .unwrap()
                    .set_rotation(rot.into());
                Ok(())
            },
        )
        .expect("Failed to set transform-component.position function impl");

    let id = Uuid::now_v7();
    let resource_table = ResourceTable::new();
    let mut script_component_instance_store = wasmtime::Store::new(
        &scripting_engine.engine,
        ScriptComponentInstanceState {
            id,
            update_subscriptions_ref: scripting_engine.update_subscriptions.clone(),
            delta_time_seconds: scripting_engine.delta_time_seconds.clone(),
            resource_table,
        },
    );
    let script_component_instance = linker
        .instantiate(&mut script_component_instance_store, &script_component)
        .expect("Failed to instantiate script component");
    let ep_func = script_component_instance
        .get_typed_func::<(), ()>(&mut script_component_instance_store, "entrypoint")
        .expect("no entrypoint defined?");
    ep_func
        .call(&mut script_component_instance_store, ())
        .expect("Failed to call entrypoint");
    ep_func
        .post_return(&mut script_component_instance_store)
        .expect("Failed to post-return entrypoint");
    scripting_engine.instance_map.insert(
        id,
        ScriptComponentInstance::new(script_component_instance_store, script_component_instance),
    );

    while let Some(ev) = e.event_receivers().wait_for_event().await {
        match ev {
            peridot::Event::Shutdown => break,
            peridot::Event::NextFrame => {
                let fd = match e.prepare_frame() {
                    Ok(x) => x,
                    Err(peridot::PrepareFrameError::FramebufferOutOfDate) => {
                        todo!("resize handling in NextFrame");
                    }
                };

                scripting_engine.update(fd.delta_time.as_secs_f32());

                let transform_lock = renderer.cube_transform_component.read().unwrap();
                renderer
                    .object_upload_buffer
                    .guard_map(BufferMapMode::Write, |ptr| unsafe {
                        ptr.clone_to(
                            0,
                            &ObjectUniformData {
                                object_matrix: peridot_math::Matrix4::trs(
                                    transform_lock.position.clone(),
                                    transform_lock.rotation.clone(),
                                    transform_lock.scale.clone(),
                                ),
                            },
                        )
                    })
                    .expect("Failed to update object data");
                drop(transform_lock);
                update_command_pool
                    .reset(true)
                    .expect("Failed to reset update command pool");
                unsafe {
                    update_command_buffer
                        .begin(e.graphics_device())
                        .expect("Failed to begin recording update commands")
                }
                .copy_buffer(
                    &renderer.object_upload_buffer,
                    &renderer.device_buffer,
                    &[br::BufferCopy::copy_data::<ObjectUniformData>(
                        0,
                        renderer.object_uniform_offset,
                    )],
                )
                .pipeline_barrier_2(&br::DependencyInfo::new(
                    &[br::MemoryBarrier2::new()
                        .from(
                            br::PipelineStageFlags2::COPY,
                            br::AccessFlags2::TRANSFER.write,
                        )
                        .to(
                            br::PipelineStageFlags2::VERTEX_SHADER,
                            br::AccessFlags2::UNIFORM_READ,
                        )],
                    &[],
                    &[],
                ))
                .end()
                .expect("Failed to finish command recording");

                e.do_render(
                    fd.backbuffer_index,
                    Some(br::EmptySubmissionBatch.with_command_buffers(&[update_command_buffer])),
                    br::EmptySubmissionBatch.with_command_buffers(&[
                        main_command_buffers[fd.backbuffer_index as usize]
                    ]),
                )
                .expect("Failed to render");
            }
            peridot::Event::Resize(new_size) => {
                main_command_pool
                    .reset(true)
                    .expect("Failed to reset main commands");
                drop(frame_buffers);
                drop(backbuffer_resources);

                e.resize_presenter_backbuffers(new_size);

                frame_size.width = new_size.0 as _;
                frame_size.height = new_size.1 as _;

                renderer
                    .resize(
                        e,
                        &mut memory_manager,
                        peridot_math::Vector2(new_size.0 as _, new_size.1 as _),
                    )
                    .expect("Failed to resize in renderer");

                backbuffer_resources = e.iter_back_buffers().cloned().collect::<Vec<_>>();
                frame_buffers = backbuffer_resources
                    .iter()
                    .map(|bb| {
                        br::FramebufferBuilder::new(&renderer.main_render_pass)
                            .with_attachment(bb)
                            .with_attachment(&renderer.depth_buffer)
                            .create()
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .expect("Failed to create framebuffer");

                for (cb, fb) in main_command_buffers.iter_mut().zip(frame_buffers.iter()) {
                    renderer
                        .populate_main_commands(
                            &mut unsafe { cb.synchronize_with(&mut main_command_pool) },
                            fb,
                        )
                        .expect("Failed to populate main commands");
                }
            }
        }
    }

    unsafe {
        e.graphics_device().wait().expect("Failed to wait works");
    }
}
