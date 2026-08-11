// Prevents an additional console window on Windows in release, do not remove!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    local_movie_translator_lib::run();
}
