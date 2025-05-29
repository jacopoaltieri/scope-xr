import argparse

def get_args(
    img_path,
    out_dir,
    pixel_size,
    circle_diameter,
    use_hough,
    magnification,
    min_n,
    n_angles,
    profile_half_length,
    derivative_step,
    symmetrize,
    axis_shifts,
    filter_name,
    shift_sino,
    avg_neighbors,
    show_plots,
):
    parser = argparse.ArgumentParser()

    parser.add_argument("--f", type=str, help="Path to the image file (.raw/.png/.tif)")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--p", type=float, help="Pixel size in mm")
    parser.add_argument("--d", type=float, help="Circle diameter in mm")
    parser.add_argument("--hough", action="store_true", help="Use Hough transform")
    parser.add_argument("--m", type=float, help="Magnification")  # New argument added here
    parser.add_argument("--n", type=int, help="Minimum pixel count")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--hl", type=int, help="Half profile length")
    parser.add_argument("--ds", type=int, help="Derivative step size")
    parser.add_argument("--axis_shifts", type=int, default=10, help="Number of axis shift steps")
    parser.add_argument("--filter", type=str, help="Reconstruction filter name")
    parser.add_argument("--sym", action="store_true", help="Symmetrize the sinogram")
    parser.add_argument('--show', action="store_true", help="Show plots")

    # Mutually exclusive group for shift
    shift_group = parser.add_mutually_exclusive_group()
    shift_group.add_argument("--shift", dest="shift_sino", action="store_true", help="Enable sinogram shifting")
    shift_group.add_argument("--no_shift", dest="shift_sino", action="store_false", help="Disable sinogram shifting")
    parser.set_defaults(shift_sino=shift_sino)

    # Mutually exclusive group for avg_neighbors
    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument("--avg", dest="avg_neighbors", action="store_true", help="Enable averaging neighboring profiles")
    avg_group.add_argument("--no_avg", dest="avg_neighbors", action="store_false", help="Disable averaging neighboring profiles")
    parser.set_defaults(avg_neighbors=avg_neighbors)

    args = parser.parse_args()

    return {
        "img_path": args.f if args.f is not None else img_path,
        "out_dir": args.o if args.o is not None else out_dir,
        "pixel_size": args.p if args.p is not None else pixel_size,
        "circle_diameter": args.d if args.d is not None else circle_diameter,
        "use_hough": args.hough if args.hough else use_hough,
        "magnification": args.m if args.m is not None else magnification,
        "min_n": args.n if args.n is not None else min_n,
        "n_angles": args.nangles if args.nangles is not None else n_angles,
        "profile_half_length": args.hl if args.hl is not None else profile_half_length,
        "derivative_step": args.ds if args.ds is not None else derivative_step,
        "axis_shifts": args.axis_shifts if args.axis_shifts is not None else axis_shifts,
        "filter_name": args.filter if args.filter is not None else filter_name,
        "symmetrize": args.sym if args.sym else symmetrize,
        "shift_sino": args.shift_sino,
        "avg_neighbors": args.avg_neighbors,
        "show_plots": args.show if args.show else show_plots,
    }
