VertexInput {
    Binding 0 [PerVertex] { pos: vec4; normal: vec4; }
}
Uniform[VertexShader](0, 0) CameraUniformData {
    mat4 viewProjectionMatrix;
}
Uniform[VertexShader](1, 0) ObjectUniformData {
    mat4 objectMatrix;
}
VertexShader {
    RasterPosition = pos * objectMatrix * viewProjectionMatrix;
    normal_v = normalize(normal * objectMatrix * viewProjectionMatrix);
}
Varyings VertexShader -> FragmentShader {
    normal_v: vec4;
}
FragmentShader {
    const vec3 lightIncidentDir = vec3(0.0, 0.0, 1.0);

    const float diffuse = pow(dot(-lightIncidentDir, normal_v.xyz) * 0.5 + 0.5, 2.0);
    
    Target[0] = vec4(vec3(1.0, 1.0, 1.0) * diffuse, 1.0);
}
