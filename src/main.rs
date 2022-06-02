use misori::mis_opt::Histogram;
use nalgebra::Vector3;
use rand_pcg::Pcg64;
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::vec;

use misori::*;

fn main1() {
    let bnds = parse_bnds("cubes.stface");
    let num_vols = count_volumes_from_bnds(&bnds);
    let mut g = build_graph(bnds, vec![1.0; num_vols]);
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    // let mut g = misori::parse_graph("cubes.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    // write_orientations_mtex_euler(&g, "orientations-euler.out");
    // return;

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let misang_range = mis_opt::cubic_range();
    let mut hist = mis_opt::Histogram::new(misang_range.clone(), 30);
    mis_opt::normalize_grain_boundary_area(&mut g);
    hist.add_from_slice(&mis_opt::angle_area_vec(&g));

    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
    
    let now = Instant::now();
    let distrfn = |x| lognorm.pdf(x);
    let mut swapper = mis_opt::Swapper::new_with_distr(&hist, &distrfn);
    for i in 0..2_000_000 {
        if let SwapOptResult::MoreOptimal{ criterion: error } = swapper.swap(
            misori::random_grains2(&g, &mut rng),
            &mut g, &mut hist, &syms
        ) {
            // println!("iter {}, norm {}", i, error);
        } else {
            swapper.undo(&mut g, &mut hist);
        }
    }
    println!(
        "swaps alg time: {}, norm {}", 
        now.elapsed().as_secs_f64(), mis_opt::mean_squared_error(&mut hist, |x| lognorm.pdf(x))
    );

    let mut file = File::create("hist.txt").unwrap();
    for (angle, density) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
    }

    write_orientations_mtex_euler(&g, "orientations-euler.out");
}

fn main2() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    // let mut g = misori::parse_graph("bnds-10k.stface", "vols-10k.stpoly");
    let mut g = misori::parse_graph("polyqd-12k-bnds.stface", "polyqd-12k-vols.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);
    // write_orientations_mtex_euler(&g, "orientations-euler.out");
    // return;

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let misang_range = mis_opt::cubic_range();
    let mut hist = mis_opt::Histogram::new(misang_range.clone(), 30);
    mis_opt::normalize_grain_boundary_area(&mut g);
    hist.add_from_slice(&mis_opt::angle_area_vec(&g));

    let uni = StatUniform::new(misang_range.start, misang_range.end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
    
    let now = Instant::now();
    // let distr = |x| lognorm.pdf(x);
    let dh = hist.clone().normalize();
    let distr = move |x| dh.heights[dh.heights.len() - dh.idx(x) - 1] / dh.bar_len;
    // let mut swapper = mis_opt::Swapper::new_with_distr(&hist,  &distr);
    let cend = mis_opt::cubic_range().end;
    // let loss_fn = Box::new(|h: &Histogram| h.pairs().map(|(a, d)| (cend - a) * (cend - a) * d * d).sum());
    let loss_fn = Box::new(|h: &Histogram| h.pairs().map(|(a, d)| a * a * d * d).sum());
    let mut swapper = mis_opt::Swapper::new_with_loss_fn(&hist, loss_fn);

    let mut temphist = hist.clone();
    let temphistlen = temphist.heights.len(); 
    // for i in 0..temphistlen {
    //     temphist.heights[temphistlen - i - 1] = hist.heights[i]
    // }
    let mut file = File::create("hist.txt").unwrap();
    for (angle, density) in temphist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
    }
    // return;
    
    for i in 0..5_000_000 {
        if let SwapOptResult::MoreOptimal{ criterion: error, .. } = swapper.swap(
            misori::random_grains2(&g, &mut rng),
            &mut g, &mut hist, &syms,
        ) {
            // println!("iter {}, norm {}", i, error);
        } else {
            swapper.undo(&mut g, &mut hist);
        }
    }
    println!(
        "swaps alg time: {}, error {}", 
        now.elapsed().as_secs_f64(), mis_opt::mean_squared_error(&mut hist, |x| lognorm.pdf(x))
    );

    let mut file = File::create("hist.txt").unwrap();
    for (angle, density) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
    }

    write_orientations_mtex_euler(&g, "orientations-euler.out");
}

fn main() {
    main2();

    // let center = Vector3::new(0.4, 0.3, 0.0);
    // let radius = 0.03;
    // let num_oris = 10000;
    // write_random_orientations_in_rodrigues_sphere_mtex_euler(
    //     num_oris, center, radius, "orientations-euler.out"
    // );
    // write_random_orientations_in_homochoric_sphere_mtex_euler(
    //     num_oris, center, radius, "orientations-euler.out"
    // );

    // write_random_orientations_mtex_euler(num_oris, "orientations-euler.out");
    // write_cells_center_orientations_mtex_euler(10, "orientations-euler.out");
    // write_cells_random_orientations_mtex_euler(10, "orientations-euler.out");
    // write_cell_random_orientations_mtex_euler(5, "orientations-euler.out");
    // write_cells_with_growing_disturbance_orientations_mtex_euler(20, "orientations-euler.out");
    // write_diagonal_cells_center_orientations_mtex_euler(100, "orientations-euler.out");
    // write_diagonal_cells_with_growing_disturbance_orientations_mtex_euler(100, "orientations-euler.out");
}
