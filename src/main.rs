use rand_pcg::Pcg64;
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use misori::*;

fn main1() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut g = misori::parse_graph("bnds-10k.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);
    // write_orientations_mtex_euler(&g, "orientations-euler.out");

    let segms = 17;
    let mut grid = fnd::FundGrid::new(segms);
    ori_opt::normalize_grain_volumes(&mut g);
    grid.add_from_iter(g.node_weights());

    let minmax_density = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten()
            .min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol, 
        *g.cells.iter().flatten().flatten()
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol
    );

    println!("min max ori density: {:?}", minmax_density(&grid));
    
    let now = Instant::now();
    let mut rotator = ori_opt::Rotator::new(&grid);
    println!("starting dnorm: {}", rotator.quad_dnorm.sqrt().sqrt());
    for i in 0..10_000_000 {
        if let RotationOptResult::MoreOptimal{ criterion: dnorm, .. } = rotator.rotate(
            RotationMode::Start,
            misori::random_grain(&g, &mut rng),
            &mut g, &mut grid, &mut rng
        ) {
            // println!("iter {}, dnorm {}", i, dnorm);
        } else {
            rotator.undo(&mut g, &mut grid);
        }
    }
    println!(
        "rotations alg time: {} s, dnorm {}", 
        now.elapsed().as_secs_f64(), rotator.quad_dnorm.sqrt().sqrt()
    );

    println!("min max ori density: {:?}", minmax_density(&grid));
    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // return;

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = mis_opt::Histogram::new(hist_beg, hist_end, 30);
    mis_opt::normalize_grain_boundary_area(&mut g);
    let aa = mis_opt::angle_area_vec(&g);
    hist.add_from_slice(&aa);

    let uni = StatUniform::new(hist_beg, hist_end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
    
    let now = Instant::now();
    let mut swapper = mis_opt::Swapper::new(&hist, |x| lognorm.pdf(x));
    for i in 0..1_000_000 {
        if let SwapOptResult::MoreOptimal(dnorm) = swapper.swap(
            misori::random_grains2(&g, &mut rng),
            &mut g, &mut hist, &syms
        ) {
            // println!("iter {}, norm {}", i, dnorm);
        } else {
            swapper.undo(&mut g, &mut hist);
        }
    }
    println!(
        "swaps alg time: {}, norm {}", 
        now.elapsed().as_secs_f64(), mis_opt::diff_norm(&mut hist, |x| lognorm.pdf(x))
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
    let mut g = misori::parse_graph("bnds-10k.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);
    write_orientations_mtex_euler(&g, "orientations-euler.out");

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = mis_opt::Histogram::new(hist_beg, hist_end, 30);
    mis_opt::normalize_grain_boundary_area(&mut g);
    let aa = mis_opt::angle_area_vec(&g);
    hist.add_from_slice(&aa);

    let uni = StatUniform::new(hist_beg, hist_end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
    
    // let now = Instant::now();
    // for i in 0..1_000_000 {
    //     if let Some(dnorm) = iterate_swaps(
    //         &mut g, &mut hist, &syms, &mut rng, |x| lognorm.pdf(x)
    //     ) {
    //         // println!("iter {}, norm {}", i, dnorm);
    //     }
    // }
    // println!(
    //     "swaps alg time:        {}, norm {}", 
    //     now.elapsed().as_secs_f64(), diff_norm(&mut hist, |x| lognorm.pdf(x))
    // );

    let now = Instant::now();
    let mut rotator = mis_opt::Rotator::new();
    for i in 0..3_000_000 {
        if let RotationOptResult::MoreOptimal{ criterion: dnorm, .. } = rotator.rotate(
            RotationMode::Start,
            misori::random_grain(&g, &mut rng),
            &mut g, &mut hist, &syms, |x| lognorm.pdf(x), &mut rng
        ) {
            // println!("iter {}, norm {}", i, dnorm);
        } else {
            rotator.undo(&mut g, &mut hist);
        }
    }
    println!(
        "rotations alg time: {}, dnorm {}", 
        now.elapsed().as_secs_f64(), mis_opt::diff_norm(&mut hist, |x| lognorm.pdf(x))
    );

    let mut file = File::create("hist.txt").unwrap();
    for (angle, density) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
    }

    // write_orientations_mtex_euler(&g, "orientations-euler.out");
}

fn main3() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let segms = 21;
    let mut grid = fnd::FundGrid::new(segms);
    ori_opt::normalize_grain_volumes(&mut g);
    grid.add_from_iter(g.node_weights());

    let minmax_density = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten()
            .min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol, 
        *g.cells.iter().flatten().flatten()
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol
    );

    println!("min max ori density: {:?}", minmax_density(&grid));
    
    let now = Instant::now();
    let mut rotator = ori_opt::Rotator::new(&grid);
    println!("starting dnorm: {}", rotator.quad_dnorm.sqrt().sqrt());
    for i in 0..10_000_000 {
        if let RotationOptResult::MoreOptimal{ criterion: dnorm, .. } = rotator.rotate(
            RotationMode::Start,
            misori::random_grain(&g, &mut rng),
            &mut g, &mut grid, &mut rng
        ) {
            // println!("iter {}, dnorm {}", i, dnorm);
        } else {
            rotator.undo(&mut g, &mut grid);
        }
    }
    println!(
        "rotations alg time: {} s, dnorm {}", 
        now.elapsed().as_secs_f64(), rotator.quad_dnorm.sqrt().sqrt()
    );

    println!("min max ori density: {:?}", minmax_density(&grid));

    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_mtex_euler(&g, "orientations-euler.out", &mut rng);
}

fn main() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let segms = 17;
    let mut grid = fnd::FundGrid::new(segms);
    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = mis_opt::Histogram::new(hist_beg, hist_end, 30);
    ori_opt::normalize_grain_volumes(&mut g);
    mis_opt::normalize_grain_boundary_area(&mut g);
    grid.add_from_iter(g.node_weights());
    let aa = mis_opt::angle_area_vec(&g);
    hist.add_from_slice(&aa);

    let uni = StatUniform::new(hist_beg, hist_end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();

    let minmax_density = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten().
            min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol, 
        *g.cells.iter().flatten().flatten().
            max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol,
    );
    
    let now = Instant::now();
    let mut rotator_ori = ori_opt::Rotator::new(&grid);
    let mut rotator_mis = mis_opt::Rotator::new();

    let quad_sum = |a: f64, b: f64| a.powi(8) + 60.0 * b.powi(8);

    let mut min_crit = quad_sum(
        rotator_ori.quad_dnorm.sqrt().sqrt(),
        rotator_mis.dnorm,
    );
    for i in 0..10_000_000 {
        let grain_idx = misori::random_grain(&g, &mut rng);

        let (dnorm_ori, prev_ori) = match rotator_ori.rotate(
            RotationMode::Start,
            grain_idx, &mut g, &mut grid, &mut rng
        ) {
            RotationOptResult::MoreOptimal{ 
                criterion: dnorm, prev_ori 
            } => (dnorm, prev_ori.unwrap()),
            RotationOptResult::SameOrLessOptimal{
                criterion: dnorm, prev_ori 
            } => (dnorm, prev_ori.unwrap()),
        };
        let dnorm_mis = match rotator_mis.rotate(
            RotationMode::Continue{ prev_ori },
            grain_idx, &mut g, &mut hist, &syms, |x| lognorm.pdf(x), &mut rng
        ) {
            RotationOptResult::MoreOptimal{ criterion: dnorm, .. } => dnorm,
            RotationOptResult::SameOrLessOptimal{ criterion: dnorm, .. } => dnorm,
        };

        let crit = quad_sum(dnorm_ori, dnorm_mis);
        if crit < min_crit {
            min_crit = crit;
            // println!(
            //     "iter {}, ori dnorm {}, mis dnorm {}", 
            //     i, dnorm_ori, dnorm_mis
            // );
        } else {
            rotator_ori.undo(&mut g, &mut grid);
            rotator_mis.undo(&mut g, &mut hist);
        }
    }
    println!(
        "rotations alg time: {} s, ori dnorm {}, mis dnorm {}", 
        now.elapsed().as_secs_f64(), rotator_ori.quad_dnorm.sqrt().sqrt(), rotator_mis.dnorm
    );

    println!("min max ori density: {:?}", minmax_density(&grid));

    let mut file = File::create("hist.txt").unwrap();
    for (angle, density) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
    }
    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_mtex_euler(&g, "orientations-euler.out", &mut rng);
}
