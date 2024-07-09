use std::rc::Rc;

use bedrock::{
    self as br, CommandBuffer, CommandPool, DescriptorPool, Device, GraphicsPipelineBuilder, Image,
    ImageChild, ImageSubresourceSlice, SubmissionBatch,
};
use peridot::{EngineEvents, FeatureRequests, NativeLinker};
use peridot_command_object::{
    BeginRenderPass, BindGraphicsDescriptorSets, EndRenderPass, GraphicsCommand,
    GraphicsCommandCombiner, RangedBuffer, StandardIndexedMesh,
};
use peridot_math::{One, Zero};
use peridot_memory_manager::{BufferMapMode, MemoryManager};
use peridot_vertex_processing_pack::PvpShaderModules;

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

pub struct Renderer {
    main_render_pass: br::RenderPassObject<peridot::DeviceObject>,
    depth_buffer: Rc<br::ImageViewObject<peridot_memory_manager::Image>>,
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
    cube_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        br::PipelineLayoutObject<peridot::DeviceObject>,
    >,
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

        let dsl_ub1 = br::DescriptorSetLayoutBuilder::new()
            .bind(
                br::DescriptorType::UniformBuffer
                    .make_binding(1)
                    .only_for_vertex(),
            )
            .create(e.graphics_device().clone())?;

        let shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.plain").expect("Failed to load shader"),
        )?;
        let cube_pl = br::PipelineLayoutBuilder::new(vec![&dsl_ub1, &dsl_ub1], vec![])
            .create(e.graphics_device().clone())?;
        let mut cube_pipeline = br::NonDerivedGraphicsPipelineBuilder::new(
            &cube_pl,
            (&main_render_pass, 0),
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
        let cube_pipeline = peridot::LayoutedPipeline::combine(
            cube_pipeline.create(
                e.graphics_device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )?,
            cube_pl,
        );

        let mut dp = br::DescriptorPoolBuilder::new(2)
            .with_reservations(vec![br::DescriptorType::UniformBuffer.with_count(2)])
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
            main_render_pass,
            depth_buffer: Rc::new(depth_buffer),
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
            cube_pipeline,
        })
    }

    fn resize(
        &mut self,
        e: &mut peridot::Engine<impl NativeLinker>,
        memory_manager: &mut MemoryManager,
        new_size: peridot_math::Vector2<u32>,
    ) -> br::Result<()> {
        let rect = br::vk::VkExtent2D::from(new_size).into_rect(br::vk::VkOffset2D::ZERO);
        let viewport = rect.make_viewport(0.0..1.0);

        self.depth_buffer = Rc::new(
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
            self.cube_pipeline.layout(),
            (&self.main_render_pass, 0),
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
        unsafe {
            self.cube_pipeline.replace_pipeline(cube_pipeline.create(
                e.graphics_device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )?);
        }

        Ok(())
    }

    fn populate_main_commands(
        &self,
        cb: &mut br::SynchronizedCommandBuffer<
            impl br::CommandPool<ConcreteDevice = peridot::DeviceObject> + br::VkHandleMut,
            impl br::CommandBuffer + br::VkHandleMut,
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
                &self.cube_pipeline,
                BindGraphicsDescriptorSets::new(&[
                    self.camera_descriptor_set.0,
                    self.object_descriptor_set.0,
                ]),
            ))
            .between(
                BeginRenderPass::for_entire_framebuffer(&self.main_render_pass, fb)
                    .with_clear_values(vec![
                        br::ClearValue::color_f32([0.0, 0.0, 0.2, 1.0]),
                        br::ClearValue::depth_stencil(1.0, 0),
                    ]),
                EndRenderPass,
            )
            .execute_and_finish(cb.begin()?.as_dyn_ref())
    }
}

pub struct Game<NL: NativeLinker> {
    memory_manager: MemoryManager,
    renderer: Renderer,
    frame_size: peridot_math::Vector2<u32>,
    frame_buffers: Vec<br::FramebufferObject<'static, peridot::DeviceObject>>,
    main_command_pool: br::CommandPoolObject<peridot::DeviceObject>,
    main_command_buffers: Vec<br::CommandBufferObject<peridot::DeviceObject>>,
    update_command_pool: br::CommandPoolObject<peridot::DeviceObject>,
    update_command_buffer: br::CommandBufferObject<peridot::DeviceObject>,
    _ph: core::marker::PhantomData<*const NL>,
}
impl<NL: NativeLinker> FeatureRequests for Game<NL> {}
impl<NL: NativeLinker> EngineEvents<NL> for Game<NL> {
    fn init(e: &mut peridot::Engine<NL>) -> Self {
        let mut memory_manager = MemoryManager::new(e.graphics());
        let renderer =
            Renderer::new(e, &mut memory_manager).expect("Failed to initialize renderer");

        let frame_size = e.back_buffer(0).unwrap().image().size().as_2d_ref().clone();
        let frame_buffers = e
            .iter_back_buffers()
            .map(|bb| {
                br::FramebufferBuilder::new(&renderer.main_render_pass)
                    .with_attachment(bb.clone())
                    .with_attachment(renderer.depth_buffer.clone())
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
        let [update_command_buffer] = update_command_pool
            .alloc_array::<1>(true)
            .expect("Failed to allocate update command buffer");

        Self {
            memory_manager,
            renderer,
            frame_size: frame_size.into(),
            frame_buffers,
            main_command_pool,
            main_command_buffers,
            update_command_pool,
            update_command_buffer,
            _ph: core::marker::PhantomData,
        }
    }

    fn update(
        &mut self,
        e: &mut peridot::Engine<NL>,
        on_back_buffer_of: u32,
        _delta_time: std::time::Duration,
    ) {
        self.renderer
            .object_upload_buffer
            .guard_map(BufferMapMode::Write, |ptr| unsafe {
                ptr.clone_to(
                    0,
                    &ObjectUniformData {
                        object_matrix: peridot_math::Matrix4::ONE,
                    },
                )
            })
            .expect("Failed to update object data");
        self.update_command_pool
            .reset(true)
            .expect("Failed to reset update command pool");
        unsafe {
            self.update_command_buffer
                .begin(e.graphics_device())
                .expect("Failed to begin recording update commands")
        }
        .copy_buffer(
            &self.renderer.object_upload_buffer,
            &self.renderer.device_buffer,
            &[br::BufferCopy::copy_data::<ObjectUniformData>(
                0,
                self.renderer.object_uniform_offset,
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
            on_back_buffer_of,
            Some(br::EmptySubmissionBatch.with_command_buffers(&[self.update_command_buffer])),
            br::EmptySubmissionBatch
                .with_command_buffers(&[self.main_command_buffers[on_back_buffer_of as usize]]),
        )
        .expect("Failed to render");
    }

    fn discard_back_buffer_resources(&mut self) {
        self.main_command_pool
            .reset(true)
            .expect("Failed to reset main commands");
        self.frame_buffers.clear();
    }

    fn on_resize(&mut self, e: &mut peridot::Engine<NL>, new_size: peridot_math::Vector2<usize>) {
        self.frame_size = peridot_math::Vector2(new_size.0 as _, new_size.1 as _);

        self.renderer
            .resize(
                e,
                &mut self.memory_manager,
                peridot_math::Vector2(new_size.0 as _, new_size.1 as _),
            )
            .expect("Failed to resize in renderer");

        self.frame_buffers = e
            .iter_back_buffers()
            .map(|bb| {
                br::FramebufferBuilder::new(&self.renderer.main_render_pass)
                    .with_attachment(bb.clone())
                    .with_attachment(self.renderer.depth_buffer.clone())
                    .create()
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create framebuffer");

        for (cb, fb) in self
            .main_command_buffers
            .iter_mut()
            .zip(self.frame_buffers.iter())
        {
            self.renderer
                .populate_main_commands(
                    &mut unsafe { cb.synchronize_with(&mut self.main_command_pool) },
                    fb,
                )
                .expect("Failed to populate main commands");
        }
    }
}
