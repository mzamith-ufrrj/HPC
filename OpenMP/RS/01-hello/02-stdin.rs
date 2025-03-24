//rustc -C opt-level=2 01-hello-omp.rs -o hello-2
use std::process;
use std::io;
fn main() {
    // Statements here are executed when the compiled binary is called.

    // Print text to the console.
    println!("Working with input keyboard or stdin");
    let mut txt = String::new();
    let _value:Option<i32> = Some(5);
    io::stdin()
        .read_line(&mut txt)
        .expect("Failure to read line");
    println!("STDIN: {}", txt);
    process::exit(0);
}
