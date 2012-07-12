import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


def calc_fractal_opencl((x1, x2, y1, y2), (h,w), maxiter):
    xx = np.arange(x1, x2, (x2-x1)/w)
    yy = np.arange(y2, y1, (y1-y2)/h) * 1j
    q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)

    output = np.empty(q.shape, dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(ctx, r"""
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, real = 0;
        float imag = 0;
        output[gid] = 0;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;

            if (real*real + imag*imag > 4.0f)
                 output[gid] = curiter;
        }
    }
    """).build()

    prg.mandelbrot(queue, output.shape, None, q_opencl,
            output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

def calc_fractal_opencl_2((x1,x2,y1,y2), (h,w), maxiter):
    output = np.empty(h*w, dtype=np.uint16)

    mf = cl.mem_flags
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
    prg = cl.Program(ctx,r"""
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(float x1, float x2, float y1, float y2,
                     int h, int w,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, real = 0;
        float imag = 0;
        float dx = (x2 - x1) / w;
        float dy = (y2 - y1) / h;
        float qx = (gid % w)*dx + x1;
        float qy = y2 - (gid / w)*dy;

        output[gid] = 0;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + qx;
            imag = 2* real*imag + qy;
            real = nreal;

            if (real*real + imag*imag > 4.0f){
                 output[gid] = curiter;
                 break;
             }
        }
    }
    """).build()

    prg.mandelbrot(queue, output.shape, None, np.float32(x1), np.float32(x2), np.float32(y1), np.float32(y2), np.int32(h), np.int32(w),
            output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output
