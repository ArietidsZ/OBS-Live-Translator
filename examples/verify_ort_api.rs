use ort::session::Session;

fn main() {
    println!("Verifying ort API...");
    // Just checking if we can reference the types
    let _builder = Session::builder().unwrap();
    println!("Session builder created successfully");
}
