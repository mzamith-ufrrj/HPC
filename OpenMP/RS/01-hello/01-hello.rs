//rustc -C opt-level=2 01-hello-omp.rs -o hello-2
use std::process;
fn main() {
    // Statements here are executed when the compiled binary is called.

    // Print text to the console.
    println!("Hello World!");
    process::exit(0);
}
