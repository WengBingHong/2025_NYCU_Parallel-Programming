#define BW 16
#define BH 16
#define RADIUS_MAX 3
#define PITCH (BW + 2 * RADIUS_MAX)
#define TILE_SIZE (PITCH * (BH + 2 * RADIUS_MAX))

inline float conv3_fast(__global const float *img, int w, int idx, __constant float *f)
{
    int up  = idx - w;
    int mid = idx;
    int dn  = idx + w;

    return img[up - 1] * f[0] + img[up] * f[1] + img[up + 1] * f[2] +
           img[mid - 1] * f[3] + img[mid] * f[4] + img[mid + 1] * f[5] +
           img[dn - 1] * f[6] + img[dn] * f[7] + img[dn + 1] * f[8];
}


inline void load_tile(__local float *tile,
                      __global const float *img,
                      int W, int H,
                      int ox, int oy,
                      int lx, int ly,
                      int tile_w, int tile_h)
{
    for (int ty = ly; ty < tile_h; ty += BH)
    {
        int gy = oy + ty;
        for (int tx = lx; tx < tile_w; tx += BW)
        {
            int gx = ox + tx;
            float v = 0.0f;

            if (gx >= 0 && gx < W && gy >= 0 && gy < H)
                v = img[gy * W + gx];

            tile[ty * PITCH + tx] = v;
        }
    }
}

inline float conv5_unroll(__local float *t, int cx, int cy, __constant float *f)
{
    float s = 0.0f;
    int base = (cy - 2) * PITCH + (cx - 2);

    for (int r = 0; r < 5; r++)
    {
        int tb = base + r * PITCH;
        int fb = r * 5;

        s += t[tb + 0] * f[fb + 0];
        s += t[tb + 1] * f[fb + 1];
        s += t[tb + 2] * f[fb + 2];
        s += t[tb + 3] * f[fb + 3];
        s += t[tb + 4] * f[fb + 4];
    }
    return s;
}

inline float conv7_unroll(__local float *t, int cx, int cy, __constant float *f)
{
    float s = 0.0f;
    int base = (cy - 3) * PITCH + (cx - 3);

    for (int r = 0; r < 7; r++)
    {
        int tb = base + r * PITCH;
        int fb = r * 7;

        s += t[tb + 0] * f[fb + 0];
        s += t[tb + 1] * f[fb + 1];
        s += t[tb + 2] * f[fb + 2];
        s += t[tb + 3] * f[fb + 3];
        s += t[tb + 4] * f[fb + 4];
        s += t[tb + 5] * f[fb + 5];
        s += t[tb + 6] * f[fb + 6];
    }
    return s;
}

__kernel void convolution(int FW,
                          __constant float *filter,
                          int H,
                          int W,
                          __global const float *img,
                          __global float *out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gx = get_group_id(0);
    int gy = get_group_id(1);

    if (FW == 3)
    {
        if (x > 0 && x < W - 1 && y > 0 && y < H - 1)
            out[y * W + x] = conv3_fast(img, W, y * W + x, filter);
        else
        {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++)
            {
                int yy = y + dy;
                for (int dx = -1; dx <= 1; dx++)
                {
                    int xx = x + dx;
                    if (xx >= 0 && xx < W && yy >= 0 && yy < H)
                    {
                        int fi = (dy + 1) * 3 + (dx + 1);
                        sum += img[yy * W + xx] * filter[fi];
                    }
                }
            }
            out[y * W + x] = sum;
        }
        return;
    }

    __local float tile[TILE_SIZE];

    int R = FW / 2;
    int tile_w = BW + 2 * R;
    int tile_h = BH + 2 * R;

    int ox = gx * BW - R;
    int oy = gy * BH - R;

    load_tile(tile, img, W, H, ox, oy, lx, ly, tile_w, tile_h);
    barrier(CLK_LOCAL_MEM_FENCE);

    int cx = lx + R;
    int cy = ly + R;

    float res = (FW == 5)
              ? conv5_unroll(tile, cx, cy, filter)
              : conv7_unroll(tile, cx, cy, filter);

    out[y * W + x] = res;
}
