use std::env;
use std::fs::File;
use std::io::{BufReader, Read, BufWriter, Write};
use std::convert::TryInto;
//use bytemuck::cast_slice;

struct StMatrix{
   n:u32,
   m:u32,
   v:Vec<f64>,
}
fn print_matrix(mat:&StMatrix){
    for i in 0..mat.n {
        for j in 0..mat.m {
            let k:usize = (i * mat.m + j).try_into().expect("u32 too big for usize");
            let value:f64 = mat.v[k];
            print!(" {:3.5}", value );
        }//for j in 0::mat.m {
        println!("");
    }//for i in 0::matA.n {
}

fn save_matrix(filename: String, mat:&StMatrix) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&mat.n.to_le_bytes())?;
    writer.write_all(&mat.m.to_le_bytes())?;

     //let bytes = bytemuck::cast_slice(mat.v);
    //writer.write_all(bytes)?;

    for v in &mat.v{
        writer.write_all(&v.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())

}

fn load_matrix(filename: String) -> Result<StMatrix, Box<dyn std::error::Error>>{
    //----------------------------------------------------------
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    reader.read_to_end(&mut buffer)?;

    let bytes = &buffer[0..4];
    let ln = u32::from_le_bytes(bytes.try_into().unwrap());

    let bytes = &buffer[4..8];
    let lm = u32::from_le_bytes(bytes.try_into().unwrap());


    let len:usize = (lm * ln).try_into().expect("u32 too big for usize");

    let mut mat = StMatrix{n:ln, m:lm, v:vec![0.0; len],};
    let mut index:usize = 0;

    let bytes = &buffer[8..buffer.len()];
    for chunk in bytes.chunks_exact(8){
        let f_bytes:[u8; 8] = chunk.try_into().unwrap();
        let value = f64::from_le_bytes(f_bytes);
        mat.v[index] = value;
        index += 1;
    }

    Ok(mat)
//----------------------------------------------------------
}
fn matrix_mul (mat_c: &mut StMatrix,
               mat_a: &StMatrix,
               mat_b: &StMatrix){

    for j in 0..mat_c.n {
        for i in 0..mat_c.m {
            let mut c:f64 = 0.0;
            for ja in 0..mat_a.m {
                let ak:usize = (j * mat_a.m + ja).try_into().expect("u32 too big for usize");
                let bk:usize = (ja * mat_b.m + i).try_into().expect("u32 too big for usize");
                c += mat_a.v[ak] * mat_b.v[bk];
            }//for ja in 0..mat_a.m {
            let k:usize = (j * mat_c.m + i).try_into().expect("u32 too big for usize");
            mat_c.v[k] = c;
        }//for j in 0..mat_c.m {
    }//for i in 0..mat_c.n {
}
//Unrecoverable and  recoverable erros
//unwarp return the correct value even-if it is an error which return the msg
fn main () -> Result<(), Box<dyn std::error::Error>> {
    let args:Vec<String> = env::args().collect();

    if args.len() < 4 {
        return Err("Matrizes não definidas".into());
    }

    let file_mat_a:&str = &args[1];
    let file_mat_b:&str = &args[2];
    let file_mat_c:&str = &args[3];


    println!("Multiplicação de matrizes");
    println!(" Matriz A: [{}]", file_mat_a);
    println!(" Matriz B: [{}]", file_mat_b);
    println!(" Matriz C: [{}]", file_mat_c);
    let mat_a:StMatrix = load_matrix(file_mat_a.to_string())?;
    let mat_b:StMatrix = load_matrix(file_mat_b.to_string())?;

    let lm = mat_a.n;
    let ln = mat_b.m;
    let len:usize = (lm * ln).try_into().expect("u32 too big for usize");
    let mut mat_c = StMatrix{n:ln, m:lm, v:vec![0.0; len],};
    matrix_mul(&mut mat_c, &mat_a, &mat_b);

    save_matrix(file_mat_c.to_string(), &mat_c)?;

    //print_matrix(&mat_a);
    //println!("-------------------------------------------");
    //print_matrix(&mat_b);
    //println!("-------------------------------------------");
    //print_matrix(&mat_c);
    //println!("-------------------------------------------");

    Ok(())
}
