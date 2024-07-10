use bindings::peridot::core::engine;

#[allow(warnings)]
mod bindings;

const fn q_from_rpc(q: bindings::peridot::core::engine::Quaternion) -> peridot_math::QuaternionF32 {
    peridot_math::Quaternion(q.x, q.y, q.z, q.w)
}

const fn q_to_rpc(q: peridot_math::QuaternionF32) -> bindings::peridot::core::engine::Quaternion {
    bindings::peridot::core::engine::Quaternion {
        x: q.0,
        y: q.1,
        z: q.2,
        w: q.3
    }
}

struct Component;
impl bindings::Guest for Component {
    fn entrypoint() {
        engine::log("hello from script!");
        engine::subscribe_update(1);
    }

    fn on_message(id: u32) {
        if id != 1 {
            // update以外はいったん無視
            return;
        }

        let dt = engine::delta_time_seconds();
        engine::log(&format!("component msg: {id} {dt} (approx. {} fps) {:?}", 1.0 / dt, engine::cube_transform().position()));
        
        let rot = q_from_rpc(engine::cube_transform().rotation()) * peridot_math::Quaternion::new(45.0f32.to_radians() * dt, peridot_math::Vector3::up());
        engine::cube_transform().set_rotation(q_to_rpc(rot));
    }
}

bindings::export!(Component with_types_in bindings);
