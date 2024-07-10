// Generated by `wit-bindgen` 0.25.0. DO NOT EDIT!
// Options used:
#[doc(hidden)]
#[allow(non_snake_case)]
pub unsafe fn _export_entrypoint_cabi<T: Guest>() {
    #[cfg(target_arch = "wasm32")]
    _rt::run_ctors_once();
    T::entrypoint();
}
#[doc(hidden)]
#[allow(non_snake_case)]
pub unsafe fn _export_on_message_cabi<T: Guest>(arg0: i32) {
    #[cfg(target_arch = "wasm32")]
    _rt::run_ctors_once();
    T::on_message(arg0 as u32);
}
pub trait Guest {
    fn entrypoint();
    fn on_message(id: u32);
}
#[doc(hidden)]

macro_rules! __export_world_component_script_cabi{
  ($ty:ident with_types_in $($path_to_types:tt)*) => (const _: () = {

    #[export_name = "entrypoint"]
    unsafe extern "C" fn export_entrypoint() {
      $($path_to_types)*::_export_entrypoint_cabi::<$ty>()
    }
    #[export_name = "on-message"]
    unsafe extern "C" fn export_on_message(arg0: i32,) {
      $($path_to_types)*::_export_on_message_cabi::<$ty>(arg0)
    }
  };);
}
#[doc(hidden)]
pub(crate) use __export_world_component_script_cabi;
#[allow(dead_code)]
pub mod peridot {
    #[allow(dead_code)]
    pub mod core {
        #[allow(dead_code, clippy::all)]
        pub mod math {
            #[used]
            #[doc(hidden)]
            #[cfg(target_arch = "wasm32")]
            static __FORCE_SECTION_REF: fn() =
                super::super::super::__link_custom_section_describing_imports;
            #[repr(C)]
            #[derive(Clone, Copy)]
            pub struct Vector3 {
                pub x: f32,
                pub y: f32,
                pub z: f32,
            }
            impl ::core::fmt::Debug for Vector3 {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    f.debug_struct("Vector3")
                        .field("x", &self.x)
                        .field("y", &self.y)
                        .field("z", &self.z)
                        .finish()
                }
            }
            #[repr(C)]
            #[derive(Clone, Copy)]
            pub struct Quaternion {
                pub x: f32,
                pub y: f32,
                pub z: f32,
                pub w: f32,
            }
            impl ::core::fmt::Debug for Quaternion {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    f.debug_struct("Quaternion")
                        .field("x", &self.x)
                        .field("y", &self.y)
                        .field("z", &self.z)
                        .field("w", &self.w)
                        .finish()
                }
            }
        }

        #[allow(dead_code, clippy::all)]
        pub mod engine {
            #[used]
            #[doc(hidden)]
            #[cfg(target_arch = "wasm32")]
            static __FORCE_SECTION_REF: fn() =
                super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            pub type Vector3 = super::super::super::peridot::core::math::Vector3;
            pub type Quaternion = super::super::super::peridot::core::math::Quaternion;

            #[derive(Debug)]
            #[repr(transparent)]
            pub struct TransformComponent {
                handle: _rt::Resource<TransformComponent>,
            }

            impl TransformComponent {
                #[doc(hidden)]
                pub unsafe fn from_handle(handle: u32) -> Self {
                    Self {
                        handle: _rt::Resource::from_handle(handle),
                    }
                }

                #[doc(hidden)]
                pub fn take_handle(&self) -> u32 {
                    _rt::Resource::take_handle(&self.handle)
                }

                #[doc(hidden)]
                pub fn handle(&self) -> u32 {
                    _rt::Resource::handle(&self.handle)
                }
            }

            unsafe impl _rt::WasmResource for TransformComponent {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();

                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[resource-drop]transform-component"]
                            fn drop(_: u32);
                        }

                        drop(_handle);
                    }
                }
            }

            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn position(&self) -> Vector3 {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 12]);
                        let mut ret_area = RetArea([::core::mem::MaybeUninit::uninit(); 12]);
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.position"]
                            fn wit_import(_: i32, _: *mut u8);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<f32>();
                        let l2 = *ptr0.add(4).cast::<f32>();
                        let l3 = *ptr0.add(8).cast::<f32>();
                        super::super::super::peridot::core::math::Vector3 {
                            x: l1,
                            y: l2,
                            z: l3,
                        }
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn rotation(&self) -> Quaternion {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 16]);
                        let mut ret_area = RetArea([::core::mem::MaybeUninit::uninit(); 16]);
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.rotation"]
                            fn wit_import(_: i32, _: *mut u8);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<f32>();
                        let l2 = *ptr0.add(4).cast::<f32>();
                        let l3 = *ptr0.add(8).cast::<f32>();
                        let l4 = *ptr0.add(12).cast::<f32>();
                        super::super::super::peridot::core::math::Quaternion {
                            x: l1,
                            y: l2,
                            z: l3,
                            w: l4,
                        }
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn scale(&self) -> Vector3 {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 12]);
                        let mut ret_area = RetArea([::core::mem::MaybeUninit::uninit(); 12]);
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.scale"]
                            fn wit_import(_: i32, _: *mut u8);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<f32>();
                        let l2 = *ptr0.add(4).cast::<f32>();
                        let l3 = *ptr0.add(8).cast::<f32>();
                        super::super::super::peridot::core::math::Vector3 {
                            x: l1,
                            y: l2,
                            z: l3,
                        }
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn set_position(&self, pos: Vector3) {
                    unsafe {
                        let super::super::super::peridot::core::math::Vector3 {
                            x: x0,
                            y: y0,
                            z: z0,
                        } = pos;

                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.set-position"]
                            fn wit_import(_: i32, _: f32, _: f32, _: f32);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: f32, _: f32, _: f32) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            _rt::as_f32(x0),
                            _rt::as_f32(y0),
                            _rt::as_f32(z0),
                        );
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn set_rotation(&self, rot: Quaternion) {
                    unsafe {
                        let super::super::super::peridot::core::math::Quaternion {
                            x: x0,
                            y: y0,
                            z: z0,
                            w: w0,
                        } = rot;

                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.set-rotation"]
                            fn wit_import(_: i32, _: f32, _: f32, _: f32, _: f32);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: f32, _: f32, _: f32, _: f32) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            _rt::as_f32(x0),
                            _rt::as_f32(y0),
                            _rt::as_f32(z0),
                            _rt::as_f32(w0),
                        );
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn set_scale(&self, scale: Vector3) {
                    unsafe {
                        let super::super::super::peridot::core::math::Vector3 {
                            x: x0,
                            y: y0,
                            z: z0,
                        } = scale;

                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.set-scale"]
                            fn wit_import(_: i32, _: f32, _: f32, _: f32);
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: f32, _: f32, _: f32) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            _rt::as_f32(x0),
                            _rt::as_f32(y0),
                            _rt::as_f32(z0),
                        );
                    }
                }
            }
            impl TransformComponent {
                #[allow(unused_unsafe, clippy::all)]
                pub fn set_trs(&self, pos: Vector3, rot: Quaternion, scale: Vector3) {
                    unsafe {
                        let super::super::super::peridot::core::math::Vector3 {
                            x: x0,
                            y: y0,
                            z: z0,
                        } = pos;
                        let super::super::super::peridot::core::math::Quaternion {
                            x: x1,
                            y: y1,
                            z: z1,
                            w: w1,
                        } = rot;
                        let super::super::super::peridot::core::math::Vector3 {
                            x: x2,
                            y: y2,
                            z: z2,
                        } = scale;

                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "peridot:core/engine")]
                        extern "C" {
                            #[link_name = "[method]transform-component.set-trs"]
                            fn wit_import(
                                _: i32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                                _: f32,
                            );
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(
                            _: i32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                            _: f32,
                        ) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            _rt::as_f32(x0),
                            _rt::as_f32(y0),
                            _rt::as_f32(z0),
                            _rt::as_f32(x1),
                            _rt::as_f32(y1),
                            _rt::as_f32(z1),
                            _rt::as_f32(w1),
                            _rt::as_f32(x2),
                            _rt::as_f32(y2),
                            _rt::as_f32(z2),
                        );
                    }
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            pub fn log(text: &str) {
                unsafe {
                    let vec0 = text;
                    let ptr0 = vec0.as_ptr().cast::<u8>();
                    let len0 = vec0.len();

                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "peridot:core/engine")]
                    extern "C" {
                        #[link_name = "log"]
                        fn wit_import(_: *mut u8, _: usize);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: *mut u8, _: usize) {
                        unreachable!()
                    }
                    wit_import(ptr0.cast_mut(), len0);
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            pub fn subscribe_update(id: u32) {
                unsafe {
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "peridot:core/engine")]
                    extern "C" {
                        #[link_name = "subscribe-update"]
                        fn wit_import(_: i32);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: i32) {
                        unreachable!()
                    }
                    wit_import(_rt::as_i32(&id));
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            pub fn unsubscrube_update(id: u32) {
                unsafe {
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "peridot:core/engine")]
                    extern "C" {
                        #[link_name = "unsubscrube-update"]
                        fn wit_import(_: i32);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: i32) {
                        unreachable!()
                    }
                    wit_import(_rt::as_i32(&id));
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            pub fn delta_time_seconds() -> f32 {
                unsafe {
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "peridot:core/engine")]
                    extern "C" {
                        #[link_name = "delta-time-seconds"]
                        fn wit_import() -> f32;
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import() -> f32 {
                        unreachable!()
                    }
                    let ret = wit_import();
                    ret
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            pub fn cube_transform() -> TransformComponent {
                unsafe {
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "peridot:core/engine")]
                    extern "C" {
                        #[link_name = "cube-transform"]
                        fn wit_import() -> i32;
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import() -> i32 {
                        unreachable!()
                    }
                    let ret = wit_import();
                    TransformComponent::from_handle(ret as u32)
                }
            }
        }
    }
}
mod _rt {

    use core::fmt;
    use core::marker;
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};

    /// A type which represents a component model resource, either imported or
    /// exported into this component.
    ///
    /// This is a low-level wrapper which handles the lifetime of the resource
    /// (namely this has a destructor). The `T` provided defines the component model
    /// intrinsics that this wrapper uses.
    ///
    /// One of the chief purposes of this type is to provide `Deref` implementations
    /// to access the underlying data when it is owned.
    ///
    /// This type is primarily used in generated code for exported and imported
    /// resources.
    #[repr(transparent)]
    pub struct Resource<T: WasmResource> {
        // NB: This would ideally be `u32` but it is not. The fact that this has
        // interior mutability is not exposed in the API of this type except for the
        // `take_handle` method which is supposed to in theory be private.
        //
        // This represents, almost all the time, a valid handle value. When it's
        // invalid it's stored as `u32::MAX`.
        handle: AtomicU32,
        _marker: marker::PhantomData<T>,
    }

    /// A trait which all wasm resources implement, namely providing the ability to
    /// drop a resource.
    ///
    /// This generally is implemented by generated code, not user-facing code.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait WasmResource {
        /// Invokes the `[resource-drop]...` intrinsic.
        unsafe fn drop(handle: u32);
    }

    impl<T: WasmResource> Resource<T> {
        #[doc(hidden)]
        pub unsafe fn from_handle(handle: u32) -> Self {
            debug_assert!(handle != u32::MAX);
            Self {
                handle: AtomicU32::new(handle),
                _marker: marker::PhantomData,
            }
        }

        /// Takes ownership of the handle owned by `resource`.
        ///
        /// Note that this ideally would be `into_handle` taking `Resource<T>` by
        /// ownership. The code generator does not enable that in all situations,
        /// unfortunately, so this is provided instead.
        ///
        /// Also note that `take_handle` is in theory only ever called on values
        /// owned by a generated function. For example a generated function might
        /// take `Resource<T>` as an argument but then call `take_handle` on a
        /// reference to that argument. In that sense the dynamic nature of
        /// `take_handle` should only be exposed internally to generated code, not
        /// to user code.
        #[doc(hidden)]
        pub fn take_handle(resource: &Resource<T>) -> u32 {
            resource.handle.swap(u32::MAX, Relaxed)
        }

        #[doc(hidden)]
        pub fn handle(resource: &Resource<T>) -> u32 {
            resource.handle.load(Relaxed)
        }
    }

    impl<T: WasmResource> fmt::Debug for Resource<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Resource")
                .field("handle", &self.handle)
                .finish()
        }
    }

    impl<T: WasmResource> Drop for Resource<T> {
        fn drop(&mut self) {
            unsafe {
                match self.handle.load(Relaxed) {
                    // If this handle was "taken" then don't do anything in the
                    // destructor.
                    u32::MAX => {}

                    // ... but otherwise do actually destroy it with the imported
                    // component model intrinsic as defined through `T`.
                    other => T::drop(other),
                }
            }
        }
    }

    pub fn as_f32<T: AsF32>(t: T) -> f32 {
        t.as_f32()
    }

    pub trait AsF32 {
        fn as_f32(self) -> f32;
    }

    impl<'a, T: Copy + AsF32> AsF32 for &'a T {
        fn as_f32(self) -> f32 {
            (*self).as_f32()
        }
    }

    impl AsF32 for f32 {
        #[inline]
        fn as_f32(self) -> f32 {
            self as f32
        }
    }

    pub fn as_i32<T: AsI32>(t: T) -> i32 {
        t.as_i32()
    }

    pub trait AsI32 {
        fn as_i32(self) -> i32;
    }

    impl<'a, T: Copy + AsI32> AsI32 for &'a T {
        fn as_i32(self) -> i32 {
            (*self).as_i32()
        }
    }

    impl AsI32 for i32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for u32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for i16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for u16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for i8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for u8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for char {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    impl AsI32 for usize {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn run_ctors_once() {
        wit_bindgen_rt::run_ctors_once();
    }
}

/// Generates `#[no_mangle]` functions to export the specified type as the
/// root implementation of all generated traits.
///
/// For more information see the documentation of `wit_bindgen::generate!`.
///
/// ```rust
/// # macro_rules! export{ ($($t:tt)*) => (); }
/// # trait Guest {}
/// struct MyType;
///
/// impl Guest for MyType {
///     // ...
/// }
///
/// export!(MyType);
/// ```
#[allow(unused_macros)]
#[doc(hidden)]

macro_rules! __export_component_script_impl {
  ($ty:ident) => (self::export!($ty with_types_in self););
  ($ty:ident with_types_in $($path_to_types_root:tt)*) => (
  $($path_to_types_root)*::__export_world_component_script_cabi!($ty with_types_in $($path_to_types_root)*);
  )
}
#[doc(inline)]
pub(crate) use __export_component_script_impl as export;

#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.25.0:component-script:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 939] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xa4\x06\x01A\x02\x01\
A\x0a\x01B\x04\x01r\x03\x01xv\x01yv\x01zv\x04\0\x07vector3\x03\0\0\x01r\x04\x01x\
v\x01yv\x01zv\x01wv\x04\0\x0aquaternion\x03\0\x02\x03\x01\x11peridot:core/math\x05\
\0\x02\x03\0\0\x07vector3\x02\x03\0\0\x0aquaternion\x01B\x1d\x02\x03\x02\x01\x01\
\x04\0\x07vector3\x03\0\0\x02\x03\x02\x01\x02\x04\0\x0aquaternion\x03\0\x02\x04\0\
\x13transform-component\x03\x01\x01h\x04\x01@\x01\x04self\x05\0\x01\x04\0$[metho\
d]transform-component.position\x01\x06\x01@\x01\x04self\x05\0\x03\x04\0$[method]\
transform-component.rotation\x01\x07\x04\0![method]transform-component.scale\x01\
\x06\x01@\x02\x04self\x05\x03pos\x01\x01\0\x04\0([method]transform-component.set\
-position\x01\x08\x01@\x02\x04self\x05\x03rot\x03\x01\0\x04\0([method]transform-\
component.set-rotation\x01\x09\x01@\x02\x04self\x05\x05scale\x01\x01\0\x04\0%[me\
thod]transform-component.set-scale\x01\x0a\x01@\x04\x04self\x05\x03pos\x01\x03ro\
t\x03\x05scale\x01\x01\0\x04\0#[method]transform-component.set-trs\x01\x0b\x01@\x01\
\x04texts\x01\0\x04\0\x03log\x01\x0c\x01@\x01\x02idy\x01\0\x04\0\x10subscribe-up\
date\x01\x0d\x04\0\x12unsubscrube-update\x01\x0d\x01@\0\0v\x04\0\x12delta-time-s\
econds\x01\x0e\x01i\x04\x01@\0\0\x0f\x04\0\x0ecube-transform\x01\x10\x03\x01\x13\
peridot:core/engine\x05\x03\x01@\0\x01\0\x04\0\x0aentrypoint\x01\x04\x01@\x01\x02\
idy\x01\0\x04\0\x0aon-message\x01\x05\x04\x01\x1dperidot:core/component-script\x04\
\0\x0b\x16\x01\0\x10component-script\x03\0\0\0G\x09producers\x01\x0cprocessed-by\
\x02\x0dwit-component\x070.208.1\x10wit-bindgen-rust\x060.25.0";

#[inline(never)]
#[doc(hidden)]
#[cfg(target_arch = "wasm32")]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
