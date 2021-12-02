use crate::*;
use rand_pcg::Pcg64;
use approx::{AbsDiffEq, RelativeEq};

fn test_rotate_to_fund_domain(o: UnitQuat, syms: &Vec<UnitQuat>) -> UnitQuat {
    let mut res = None;
    for s in syms {
        let q = s * o;
        let angs = EulerAngles::from(q);
        if fnd::euler_angles_inside(angs) {
            if res.is_none() {
                res = Some(q);
            } else {
                panic!(
                    "multiple orientations in fundamental domain: {:?} and {:?}", 
                    EulerAngles::from(res.unwrap()), angs
                )
            }
        }
    }
    if let Some(q) = res {
        q
    } else {
        panic!(
            "failed to rotate to fundamental domain: {:?}, {:?}", 
            o, EulerAngles::from(o)
        )
    }
}

#[test]
fn test_fund_domain() {    
    let mut rng = Pcg64::seed_from_u64(0);
    let syms = cube_rotational_symmetry();
    for &basis in &syms {
        println!("{:?}", EulerAngles::from(basis));
    }
    
    for _ in 0..100_000 {
        test_rotate_to_fund_domain(
            fnd::random_euler_angles(&mut rng).into(), 
            &syms
        );
    }
}

#[test]
fn test_quaternion_and_euler_angles_conversion() {
    use approx::assert_relative_eq;

    let mut rng = Pcg64::seed_from_u64(0);
    for _ in 0..1_000_000 {
        let angs = EulerAngles::random(&mut rng);
        let q = UnitQuat::from(angs);
        let back = EulerAngles::from(q);
        assert_relative_eq!(angs, back, max_relative = f32::EPSILON as f64);
    }
    for _ in 0..1_000 {
        let mut angs = EulerAngles::random(&mut rng);
        angs.cos_beta = 1.0;
        let mut correct_back_alpha = angs.alpha + angs.gamma;
        if correct_back_alpha >= 2.0 * PI {
            correct_back_alpha -= 2.0 * PI;
        }
        let q = UnitQuat::from(angs);
        let back = EulerAngles::from(q);
        let correct_back = EulerAngles{ alpha: correct_back_alpha, cos_beta: 1.0, gamma: 0.0 };
        assert_relative_eq!(correct_back, back, max_relative = f32::EPSILON as f64);
    }
}

#[test]
fn test_fund_angles_and_euler_angles_conversion() {
    use approx::assert_relative_eq;

    let mut rng = Pcg64::seed_from_u64(0);
    for _ in 0..1_000_000 {
        let angs = FundAngles::random(&mut rng);
        let eul = EulerAngles::from(angs);
        let back = FundAngles::from(eul);
        assert_relative_eq!(angs, back, max_relative = f32::EPSILON as f64 * 10.0);
    }
}



impl AbsDiffEq for EulerAngles {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.alpha, &other.alpha, epsilon) &&
        f64::abs_diff_eq(&self.cos_beta, &other.cos_beta, epsilon) &&
        f64::abs_diff_eq(&self.gamma, &other.gamma, epsilon)
    }
}

impl RelativeEq for EulerAngles {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self, other: &Self, 
        epsilon: Self::Epsilon, max_relative: Self::Epsilon
    ) -> bool {
        f64::relative_eq(&self.alpha, &other.alpha, epsilon, max_relative) &&
        f64::relative_eq(&self.cos_beta, &other.cos_beta, epsilon, max_relative) &&
        f64::relative_eq(&self.gamma, &other.gamma, epsilon, max_relative)
    }
}

impl AbsDiffEq for FundAngles {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.delta, &other.delta, epsilon) &&
        f64::abs_diff_eq(&self.lambda, &other.lambda, epsilon) &&
        f64::abs_diff_eq(&self.omega, &other.omega, epsilon)
    }
}

impl RelativeEq for FundAngles {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self, other: &Self, 
        epsilon: Self::Epsilon, max_relative: Self::Epsilon
    ) -> bool {
        f64::relative_eq(&self.delta, &other.delta, epsilon, max_relative) &&
        f64::relative_eq(&self.lambda, &other.lambda, epsilon, max_relative) &&
        f64::relative_eq(&self.omega, &other.omega, epsilon, max_relative)
    }
}
