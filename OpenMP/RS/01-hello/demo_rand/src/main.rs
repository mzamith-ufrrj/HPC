use std::process;
use rand::Rng;
fn main() {

    //println!("{}", value);
    for _n in 1..10000{
        let x = rand::thread_rng().gen_range(0..=100);
        println!("{}", x);
    }
    process::exit(0);
}
