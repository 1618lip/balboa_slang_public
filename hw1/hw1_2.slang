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

    image[tid] = float4(1.0);

    float2 p = float2(tid.x + 0.5, tid.y + 0.5);
    float min_distance = 1e10;

    for (int i = 0; i < num_polyline; i++) {
        if (i == num_polyline - 1 && !is_closed) break;

        float2 a = polyline[i];
        float2 b = polyline[(i + 1) % num_polyline];

        min_distance = min(min_distance, distance_to_segment(p, a, b));
    }

    // Change this to is_point_inside_polyline_non_zero(...) to use the
    // non-zero winding fill rule instead.
    bool inside = is_closed && use_fill_color
                    && is_point_inside_polyline_even_odd(p, polyline, num_polyline);

    if (inside) {
        image[tid] = float4(fill_color, 1.0);
    }

    if (min_distance <= stroke_width * 0.5 && use_stroke_color) {
        image[tid] = float4(stroke_color, 1.0);
    }
}

/* md
    $$\vec{v} = \vec{b} - \vec{a}$$
    $$\vec{w} = \vec{p} - \vec{a}$$
    $$t = \frac{\vec{w} \cdot \vec{v}}{\|\vec{v}\|^2}$$
    return $$\|\vec{p} - (\vec{a} + t\cdot\vec{v})\|$$
*/
float distance_to_segment(float2 p, float2 a, float2 b) {
    float2 v = b - a;
    float2 w = p - a;

    float t = dot(w, v) / dot(v, v);
    t = clamp(t, 0.0, 1.0);

    float2 closest = a + t * v;
    return length(p - closest);
}

/* md
For each edge from $\vec{a}$ to $\vec{b}$, check if a horizontal ray from $\vec{p}$ to the right crosses it.

**Edge crosses ray if:**
$$(a_y < p_y \leq b_y) \lor (b_y < p_y \leq a_y)$$

**If it crosses, find x-intersection:**
$$x_{int} = a_x + \frac{p_y - a_y}{b_y - a_y}(b_x - a_x)$$

**If $x_{int} > p_x$, increment crossing count.**

**Point is inside if:**
$$(\text{crossings} \bmod 2) == 1$$
*/
bool is_point_inside_polyline_even_odd(
    float2 point,
    StructuredBuffer<float2> polyline,
    int vertex_count
) {
    int ray_crossing_count = 0;

    for (int edge_index = 0; edge_index < vertex_count; edge_index++) {
        float2 edge_start = polyline[edge_index];
        float2 edge_end = polyline[(edge_index + 1) % vertex_count];

        bool edge_straddles_ray =
            (edge_start.y < point.y && point.y <= edge_end.y) ||
            (edge_end.y < point.y && point.y <= edge_start.y);

        if (edge_straddles_ray) {
            float ray_intersection_x = edge_start.x
                + (point.y - edge_start.y)
                / (edge_end.y - edge_start.y)
                * (edge_end.x - edge_start.x);

            if (point.x < ray_intersection_x) {
                ray_crossing_count++;
            }
        }
    }

    return (ray_crossing_count % 2) == 1;
}

/* md
For each edge from $\vec{a}$ to $\vec{b}$, cast a horizontal
ray from $\vec{p}$ to the right.

If the ray crosses an edge moving in the positive $y$ direction:

$$w \leftarrow w + 1$$

If it crosses an edge moving in the negative $y$ direction:

$$w \leftarrow w - 1$$

The point is inside if:

$$w \neq 0$$
*/
bool is_point_inside_polyline_non_zero(
    float2 point,
    StructuredBuffer<float2> polyline,
    int vertex_count
) {
    int winding_number = 0;

    for (int edge_index = 0; edge_index < vertex_count; edge_index++) {
        float2 edge_start = polyline[edge_index];
        float2 edge_end = polyline[(edge_index + 1) % vertex_count];

        bool crosses_in_positive_y =
            edge_start.y < point.y && point.y <= edge_end.y;
        bool crosses_in_negative_y =
            edge_end.y < point.y && point.y <= edge_start.y;

        if (crosses_in_positive_y || crosses_in_negative_y) {
            float ray_intersection_x = edge_start.x
                + (point.y - edge_start.y)
                / (edge_end.y - edge_start.y)
                * (edge_end.x - edge_start.x);

            if (point.x < ray_intersection_x) {
                winding_number += crosses_in_positive_y ? 1 : -1;
            }
        }
    }

    return winding_number != 0;
}
