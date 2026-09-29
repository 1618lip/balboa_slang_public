import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    float2 p = float2( tid.x + 0.5, height - (tid.y + 0.5) );
    PolylineHitTestResult hit = hit_test_polyline(
        p,
        polyline,
        num_polyline,
        is_closed
    );

    if (use_fill_color && hit.inside) {
        image[tid] = float4(fill_color, 1.0);
    }

    if (use_stroke_color && hit.stroke_distance <= stroke_width * 0.5) {
        image[tid] = float4(stroke_color, 1.0);
    }
}
