use misori::mis_opt::Histogram;
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
    let distrfn = |x| lognorm.pdf(x);
    let mut swapper = mis_opt::Swapper::new_with_distr(&hist, &distrfn);
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

fn main4() {
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


//

fn main() {
    let bnds = parse_bnds("cubes.stface");
    let num_vols = count_volumes_from_bnds(&bnds);

    let mut starting_hists = Vec::<Vec<(f64, f64)>>::new();
    let mut hists = Vec::<Vec<(f64, f64)>>::new();
    let mut energies = Vec::<Vec<(usize, f64)>>::new();
    let mut sucfails = Vec::<Vec<(usize, f64)>>::new();

    for seed in 0..10 {
        let mut g = build_graph(bnds.clone(), vec![1.0; num_vols]);

        let mut rng = Pcg64::seed_from_u64(seed);
        set_random_orientations(&mut g, &mut rng);
        // write_orientations_mtex_euler(&g, "orientations-euler.out");

        let segms = 17;
        let mut grid = fnd::FundGrid::new(segms);
        ori_opt::normalize_grain_volumes(&mut g);
        grid.add_from_iter(g.node_weights());
        
        let now = Instant::now();
        let mut rotator = ori_opt::Rotator::new(&grid);
        // println!("starting dnorm: {}", rotator.quad_dnorm.sqrt().sqrt());
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
        // println!(
        //     "rotations alg time: {} s, dnorm {}", 
        //     now.elapsed().as_secs_f64(), rotator.quad_dnorm.sqrt().sqrt()
        // );

        // write_orientations_mtex_euler(&g, "orientations-euler.out");
        // return;

        let syms = cube_rotational_symmetry();
        mis_opt::update_angles(&mut g, &syms);

        let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
        let mut hist = mis_opt::Histogram::new(hist_beg, hist_end, 50);
        mis_opt::normalize_grain_boundary_area(&mut g);
        let aa = mis_opt::angle_area_vec(&g);
        hist.add_from_slice(&aa);

        // let mut file = File::create("starting-hist.txt").unwrap();
        // for (angle, density) in hist.pairs() {
        //     writeln!(&mut file, "{}\t{}", angle.to_degrees(), density.to_radians()).unwrap();
        // }
        starting_hists.push(hist.pairs().collect());
        // return;

        let uni = StatUniform::new(hist_beg, hist_end).unwrap();
        let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
        
        let now = Instant::now();
        let negsumenfn = Box::new(|h: &Histogram| {
            let max_a = 15.0f64.to_radians();
            let max_e = max_a * (0.5 - max_a.ln());
            -h.pairs()
                .map(|(a, d)| if a < max_a { a * (0.5 - a.ln()) } else { max_e } * d)
                .sum::<f64>() / h.bars() as f64
        });
        let sumenfn = Box::new(|h: &Histogram| {
            let max_a = 15.0f64.to_radians();
            let max_e = max_a * (0.5 - max_a.ln());
            h.pairs()
                .map(|(a, d)| if a < max_a { a * (0.5 - a.ln()) } else { max_e } * d)
                .sum::<f64>() / h.bars() as f64
        });
        let mut swapper = mis_opt::Swapper::new_with_obj_fn(&hist, negsumenfn.clone());
        // println!("starting energy: {}", sumenfn(&hist));
        let mut energy = Vec::new();
        let energy_istep = 10_000;
        let mut sucfail = Vec::new();
        let sf_chunksize = 10_000;
        let mut sf = (0, 0);
        let maximization_until = 1_000_000;
        for i in 0..10_000_000 {
            if (i + 1) % sf_chunksize == 0 {
                sucfail.push(sf.0 as f64 / sf.1 as f64);
                sf = (0, 0);
            }
            if i == maximization_until {
                swapper = mis_opt::Swapper::new_with_obj_fn(&hist, sumenfn.clone());
            }
            if let SwapOptResult::MoreOptimal(en) = swapper.swap(
                misori::random_grains2(&g, &mut rng),
                &mut g, &mut hist, &syms
            ) {
                // println!("iter {}, norm {}", i, en);
                if i % energy_istep == 0 { energy.push(sumenfn(&hist)); }
                sf.0 += 1;
            } else {
                if i % energy_istep == 0 { energy.push(sumenfn(&hist)); }
                sf.1 += 1;
                swapper.undo(&mut g, &mut hist);
            }
        }
        println!(
            "seed: {}, swaps alg time: {}, energy {}", 
            seed, now.elapsed().as_secs_f64(), sumenfn(&hist)
        );

        energies.push(energy.iter().enumerate().map(|(i, &e)| (i * energy_istep, e)).collect());
        sucfails.push(sucfail.iter().enumerate().map(|(i, &sf)| (i * sf_chunksize, sf)).collect());
        hists.push(hist.pairs().collect());
    };

    let mut energy_file = File::create("energy.txt").unwrap();    
    for vec_i in 0..energies[0].len() {
        let mut mean = 0.0;
        for vi in 0..energies.len() {
            mean += energies[vi][vec_i].1;
        }
        mean /= sucfails.len() as f64;
        let mut twostddev = 0.0;
        for vi in 0..energies.len() {
            twostddev += (energies[vi][vec_i].1 - mean).powi(2);
        }
        twostddev /= (energies.len() - 1) as f64;
        twostddev = 2.0 * twostddev.sqrt();
        let (i, e1, e2) = (energies[0][vec_i].0, mean - twostddev, mean + twostddev);
        writeln!(&mut energy_file, "{}\t{}\t{}", i, e1, e2).unwrap();
    }
    let mut sucfail_file = File::create("sucfail.txt").unwrap();
    for vec_i in 0..sucfails[0].len() {
        let mut mean = 0.0;
        for vi in 0..sucfails.len() {
            mean += sucfails[vi][vec_i].1;
        }
        mean /= sucfails.len() as f64;
        let mut twostddev = 0.0;
        for vi in 0..sucfails.len() {
            twostddev += (sucfails[vi][vec_i].1 - mean).powi(2);
        }
        twostddev /= (sucfails.len() - 1) as f64;
        twostddev = 2.0 * twostddev.sqrt();
        let (i, sf1, sf2) = (sucfails[0][vec_i].0, mean - twostddev, mean + twostddev);
        writeln!(&mut sucfail_file, "{}\t{}\t{}", i, sf1, sf2).unwrap();
    }
    let mut file = File::create("starting-hist.txt").unwrap();
    for vec_i in 0..starting_hists[0].len() {
        let mut mean = 0.0;
        for vi in 0..starting_hists.len() {
            mean += starting_hists[vi][vec_i].1.to_radians();
        }
        mean /= sucfails.len() as f64;
        let mut twostddev = 0.0;
        for vi in 0..starting_hists.len() {
            twostddev += (starting_hists[vi][vec_i].1.to_radians() - mean).powi(2);
        }
        twostddev /= (starting_hists.len() - 1) as f64;
        twostddev = 2.0 * twostddev.sqrt();
        let (a, d1, d2) = (starting_hists[0][vec_i].0, mean - twostddev, mean + twostddev);
        writeln!(&mut file, "{}\t{}\t{}", a.to_degrees(), d1, d2).unwrap();
    }
    let mut file = File::create("hist.txt").unwrap();
    for vec_i in 0..hists[0].len() {
        let mut mean = 0.0;
        for vi in 0..hists.len() {
            mean += hists[vi][vec_i].1.to_radians();
        }
        mean /= sucfails.len() as f64;
        let mut twostddev = 0.0;
        for vi in 0..hists.len() {
            twostddev += (hists[vi][vec_i].1.to_radians() - mean).powi(2);
        }
        twostddev /= (hists.len() - 1) as f64;
        twostddev = 2.0 * twostddev.sqrt();
        let (a, d1, d2) = (hists[0][vec_i].0, mean - twostddev, mean + twostddev);
        writeln!(&mut file, "{}\t{}\t{}", a.to_degrees(), d1, d2).unwrap();
    }
}
