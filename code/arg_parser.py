import argparse

def get_args(
    img_path,
    out_dir,
    show_plots,
    use_hough,
    pixel_size,
    circle_diameter,
    min_n,
    n_angles,
    profile_half_length,
    derivative_step,
    filter_name,
    symmetrize,
    shift_sino,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument('--show', action="store_true", help="Show plots")
    parser.add_argument("--hough", action="store_true", help="Use Hough transform")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument("--n", type=int, help="Minimum pixel count")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--sym", action="store_true", help="Symmetrize the sinogram")
    parser.add_argument("--shift", action="store_true", help="Shift the sinogram")

    args = parser.parse_args()

    return {
        "img_path": args.f if args.f is not None else img_path,
        "out_dir": args.o if args.o is not None else out_dir,
        "show_plots": args.show if args.show else show_plots,
        "use_hough": args.hough if args.hough else use_hough,
        "pixel_size": args.p if args.p is not None else pixel_size,
        "circle_diameter": args.d if args.d is not None else circle_diameter,
        "min_n": args.n if args.n is not None else min_n,
        "n_angles": args.nangles if args.nangles is not None else n_angles,
        "profile_half_length": args.hl if args.hl is not None else profile_half_length,
        "derivative_step": args.ds if args.ds is not None else derivative_step,
        "filter_name": args.filter if args.filter is not None else filter_name,
        "symmetrize": args.sym if args.sym else symmetrize,
        "shift_sino": args.shift if args.shift else shift_sino,
    }