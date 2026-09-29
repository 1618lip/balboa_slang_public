// Common scene definitions
enum ShapeType {
    Circle,
    Polyline,
    QuadraticBezier
};

struct Shape {
    ShapeType type;
    // Index the shape into the type-specific arrays
    // (circles, polylines, or quadratic_beziers)
    int index;

    bool use_fill;
    float4 fill_rgba;
    bool use_stroke;
    float4 stroke_rgba;
    float stroke_width;
    // Maps screen/world-space positions to the shape's object space.
    float3x3 world_to_obj;
};

struct Circle {
    float2 center;
    float radius;
};

struct Polyline {
    int pts_begin;
    int pts_end;
    bool closed;
};

struct QuadraticBezier {
    int pts_begin;
    int pts_end;
    bool closed;
};

static const float PI = 3.14159265358979323846;

// Polynomial solvers
// Return true if a root is found.
// The two real roots are stored in t
bool solve_quadratic(float a, float b, float c,
                     out float2 t) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/pbrt.h#L419
    float discrim = b * b - 4 * a * c;
    if (discrim < 0) {
        t = float2(0, 0);
        return false;
    }
    float root_discrim = sqrt(discrim);

    float q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    t.x = q / a;
    t.y = c / q;
    return true;
}

// Returns real candidate roots needed for finding extrema.
// Repeated roots may be omitted.
int solve_cubic(float a, float b, float c, float d, out float3 t) {
    if (abs(a) < 1e-6) {
        if (solve_quadratic(b, c, d, t.xy)) {
            return 2;
        } else {
            return 0;
        }
    }
    // normalize cubic equation
    b /= a;
    c /= a;
    d /= a;
    float Q = (b * b - 3 * c) / 9.0;
    float R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.0;
    if (R * R < Q * Q * Q) {
        // 3 real roots
        float theta = acos(R / sqrt(Q * Q * Q));
        t.x = -2.0 * sqrt(Q) * cos(theta / 3.0) - b / 3.0;
        t.y = -2.0 * sqrt(Q) * cos((theta - 2.0 * PI) / 3.0) - b / 3.0;
        t.z = -2.0 * sqrt(Q) * cos((theta + 2.0 * PI) / 3.0) - b / 3.0;
        return 3;
    } else {
        float A = R > 0 ? -pow(R + sqrt(R * R - Q * Q * Q), 1.0/3.0):
                           pow(-R + sqrt(R * R - Q * Q * Q), 1.0/3.0);
        float B = abs(A) > 1e-6 ? Q / A : 0.0;
        t.x = (A + B) - b / 3.0;
        return 1;
    }
}

/*
 HELPER FUNCTIONS (common to all parts of the assignment)
*/
/* md
    $$l = \frac{(\mathbf{p}_1 - \mathbf{p}_0) \cdot
    (\mathbf{q} - \mathbf{p}_0)}{\|\mathbf{p}_1 - \mathbf{p}_0\|}$$

    $$\mathbf{q}' = \mathbf{p}_0 + l \cdot \frac{\mathbf{p}_1 - \mathbf{p}_0}{\|\mathbf{p}_1 - \mathbf{p}_0\|}$$

    If $0 \leq l \leq \|\mathbf{p}_1 - \mathbf{p}_0\|$, return
    $\|\mathbf{q} - \mathbf{q}'\|$.
*/
float distance_to_segment(float2 q, float2 p0, float2 p1) {
    float segment_length = length(p1 - p0);

    if (segment_length <= 1e-8) {
        return 1e10;
    }

    float l = dot(p1 - p0, q - p0) / segment_length;

    // Reject points beyond the segment's flat caps.
    if (l < 0.0 || l > segment_length) {
        return 1e10;
    }

    float2 q_prime = p0 + l * ((p1 - p0) / segment_length);
    return length(q - q_prime);
}

/* md
    For each edge from $\mathbf{p0}$ to $\mathbf{p1}$, check if a horizontal ray from $\mathbf{p}$ to the right crosses it.

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
    For each edge from $\mathbf{p0}$ to $\mathbf{p1}$, cast a horizontal
    ray from $\mathbf{p}$ to the right.

    If ray crosses an edge moving in:
    - $+y$ direction: $w++$
    - $-y$ direction: $w--$

    Point is inside if $w \neq 0$
*/
bool is_point_inside_polyline_non_zero(
    float2 point,
    StructuredBuffer<float2> polyline,
    int pts_begin,
    int pts_end
) {
    int winding_number = 0;
    int vertex_count = pts_end - pts_begin;

    for (int edge_index = 0; edge_index < vertex_count; edge_index++) {
        float2 edge_start = polyline[pts_begin + edge_index];
        float2 edge_end = polyline[
            pts_begin + (edge_index + 1) % vertex_count
        ];

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

// Convenience overload for a polyline that starts at index zero.
bool is_point_inside_polyline_non_zero(
    float2 point,
    StructuredBuffer<float2> polyline,
    int vertex_count
) {
    return is_point_inside_polyline_non_zero(
        point,
        polyline,
        0,
        vertex_count
    );
}

struct PolylineHitTestResult {
    bool inside;
    float stroke_distance;
};

// Tests both the fill and stroke geometry of a polyline. Stroke segments use
// flat caps, while interior vertices use round joins.
PolylineHitTestResult hit_test_polyline(
    float2 point,
    StructuredBuffer<float2> points,
    int pts_begin,
    int pts_end,
    bool closed
) {
    PolylineHitTestResult result;
    result.inside = false;
    result.stroke_distance = 1e10;

    int point_count = pts_end - pts_begin;
    if (point_count < 2) {
        return result;
    }

    if (closed && point_count >= 3) {
        result.inside = is_point_inside_polyline_non_zero(
            point,
            points,
            pts_begin,
            pts_end
        );
    }

    for (int edge_index = 0; edge_index < point_count; edge_index++) {
        if (!closed && edge_index == point_count - 1) {
            break;
        }

        int p0_index = pts_begin + edge_index;
        int p1_index = pts_begin + (edge_index + 1) % point_count;
        float2 p0 = points[p0_index];
        float2 p1 = points[p1_index];

        result.stroke_distance = min(
            result.stroke_distance,
            distance_to_segment(point, p0, p1)
        );

        // Closed polylines join every vertex. Open polylines leave their two
        // endpoints as flat caps and add round joins only at interior points.
        if (closed || edge_index > 0) {
            result.stroke_distance = min(
                result.stroke_distance,
                length(point - p0)
            );
        }
    }

    return result;
}

// Convenience overload for a standalone polyline starting at index zero.
PolylineHitTestResult hit_test_polyline(
    float2 point,
    StructuredBuffer<float2> points,
    int point_count,
    bool closed
) {
    return hit_test_polyline(point, points, 0, point_count, closed);
}

// Convenience overload for a Polyline stored in the shared points buffer.
PolylineHitTestResult hit_test_polyline(
    float2 point,
    StructuredBuffer<float2> points,
    Polyline polyline
) {
    return hit_test_polyline(
        point,
        points,
        polyline.pts_begin,
        polyline.pts_end,
        polyline.closed
    );
}
