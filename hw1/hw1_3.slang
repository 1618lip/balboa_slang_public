import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);
    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);

    // Earlier shapes in the scene are drawn in front of later shapes.
    for (int i = num_shapes - 1; i >= 0; i--) {
        Shape shape = shapes[i];

        switch (shape.type) {
            case ShapeType::Circle: {
                Circle circle = circles[shape.index];
                float distance_to_center = length(p - circle.center);

                if (shape.use_fill && distance_to_center <= circle.radius) {
                    image[tid] = shape.fill_rgba;
                }

                if (shape.use_stroke &&
                    abs(distance_to_center - circle.radius) <= shape.stroke_width * 0.5) {
                    image[tid] = shape.stroke_rgba;
                }

                break;
            }

            case ShapeType::Polyline: {
                Polyline polyline = polylines[shape.index];
                PolylineHitTestResult hit = hit_test_polyline(
                    p,
                    points,
                    polyline
                );

                if (shape.use_fill && hit.inside) {
                    image[tid] = shape.fill_rgba;
                }

                if (shape.use_stroke &&
                    hit.stroke_distance <= shape.stroke_width * 0.5) {
                    image[tid] = shape.stroke_rgba;
                }

                break;
            }

            case ShapeType::QuadraticBezier: {
                // Quadratic Bezier curves are implemented in a later part.
                break;
            }
        }
    }
}
