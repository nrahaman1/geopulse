//! GeoPulse desktop: the GeoPulse web app in a native window, plus GeoPulse's Python engine on this PC.
//!
//! As in GeoLibre, the engine is a locked Python project bundled with the app (`python/`: pyproject.toml, uv.lock and
//! the `geopulse` package) that uv installs on first use into the app's data folder: a Python interpreter and PyTorch
//! built for CUDA when an NVIDIA GPU is present (Metal on Apple silicon, CPU otherwise). The engine listens on
//! 127.0.0.1 only, on a free port, and accepts only requests that carry a per-launch token. It exits with the app,
//! whose process it watches.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

/// The largest AOI the PC engine accepts (the browser engine stops at 300 km²).
const MAX_JOB_KM2: &str = "1500";
/// The engine's Python (its PyTorch, rasterio and GDAL wheels are the most widely tested on it).
const PYTHON: &str = "3.12";

/// The newest installed CPython `PYTHON` patch release under `dir` (a real folder, not a link).
fn find_python(dir: &Path) -> Option<PathBuf> {
    let prefix = format!("cpython-{PYTHON}.");
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir() && !t.is_symlink()))
        .map(|e| {
            if cfg!(windows) {
                e.path().join("python.exe")
            } else {
                e.path().join("bin").join("python3")
            }
        })
        .filter(|py| py.is_file())
        .collect();
    found.sort();
    found.pop()
}

/// Remove uv's minor-version links (junctions on Windows, symlinks elsewhere); nothing is lost, they only alias.
fn remove_links(dir: &Path) {
    for entry in std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_ok_and(|t| t.is_symlink()) {
            let _ =
                std::fs::remove_dir(entry.path()).or_else(|_| std::fs::remove_file(entry.path()));
        }
    }
}

#[derive(Default)]
struct Engine(Mutex<Option<Running>>);

struct Running {
    child: Child,
    info: EngineInfo,
}

#[derive(Clone, Serialize)]
struct EngineInfo {
    url: String,
    token: String,
}

#[derive(Serialize)]
struct EngineStatus {
    installed: bool,
    /// Name of the GPU the engine will use, if any.
    gpu: Option<String>,
    /// PyTorch build: "gpu" (CUDA / Metal) or "cpu".
    variant: &'static str,
    /// Rough size of the one-time download.
    download: &'static str,
}

fn engine_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map(|d| d.join("engine"))
        .map_err(|e| format!("no app data folder: {e}"))
}

fn env_dir(app: &AppHandle, variant: &str) -> Result<PathBuf, String> {
    Ok(engine_dir(app)?.join(format!("env-{variant}")))
}

fn python(env: &Path) -> PathBuf {
    if cfg!(windows) {
        env.join("Scripts").join("python.exe")
    } else {
        env.join("bin").join("python")
    }
}

/// uv ships next to the app binary (Tauri `externalBin`).
fn uv() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let uv = exe.with_file_name(format!("uv{}", std::env::consts::EXE_SUFFIX));
    uv.is_file()
        .then_some(uv)
        .ok_or_else(|| format!("uv is missing next to {}", exe.display()))
}

/// Console programs started from a GUI app would each open a console window on Windows.
fn hidden(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn nvidia_gpu() -> Option<String> {
    let out = hidden(&mut Command::new("nvidia-smi"))
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    (out.status.success() && !name.is_empty()).then_some(name)
}

fn detect() -> (Option<String>, &'static str, &'static str) {
    if cfg!(target_os = "macos") {
        let gpu = cfg!(target_arch = "aarch64").then(|| "Apple silicon GPU".to_string());
        return (gpu, "gpu", "300 MB");
    }
    match nvidia_gpu() {
        Some(name) => (Some(name), "gpu", "3 GB"),
        None => (None, "cpu", "400 MB"),
    }
}

#[tauri::command]
fn engine_status(app: AppHandle) -> Result<EngineStatus, String> {
    let (gpu, variant, download) = detect();
    let installed = engine_dir(&app)?
        .join(format!("env-{variant}.ok"))
        .is_file();
    Ok(EngineStatus {
        installed,
        gpu,
        variant,
        download,
    })
}

/// Run a command, forwarding each output line to the web app as an "engine-log" event.
fn run_logged(app: &AppHandle, cmd: &mut Command) -> Result<(), String> {
    let mut child = hidden(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {:?}: {e}", cmd.get_program()))?;
    let out = forward(app, child.stdout.take());
    let err = forward(app, child.stderr.take());
    let status = child.wait().map_err(|e| e.to_string())?;
    let tail: Vec<String> = [
        out.join().unwrap_or_default(),
        err.join().unwrap_or_default(),
    ]
    .concat();
    if status.success() {
        Ok(())
    } else {
        Err(tail
            .iter()
            .rev()
            .take(6)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

/// Stream a pipe line by line to the web app (drained continuously: a full pipe would stall the child).
fn forward<R: Read + Send + 'static>(
    app: &AppHandle,
    pipe: Option<R>,
) -> std::thread::JoinHandle<Vec<String>> {
    let app = app.clone();
    std::thread::spawn(move || {
        let mut tail = Vec::new();
        if let Some(pipe) = pipe {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                let _ = app.emit("engine-log", &line);
                tail.push(line);
                if tail.len() > 50 {
                    tail.remove(0);
                }
            }
        }
        tail
    })
}

fn token() -> Result<String, String> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    Ok(listener.local_addr().map_err(|e| e.to_string())?.port())
}

fn healthy(port: u16) -> bool {
    let Ok(mut s) =
        TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(500))
    else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let mut head = [0u8; 12];
    s.write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .is_ok()
        && s.read_exact(&mut head).is_ok()
        && &head[9..12] == b"200"
}

/// Install (first time) or update the engine, then start it and wait until it answers.
#[tauri::command]
async fn engine_start(
    app: AppHandle,
    state: State<'_, Engine>,
    variant: String,
) -> Result<EngineInfo, String> {
    if let Some(running) = state.0.lock().unwrap().as_mut() {
        if running
            .child
            .try_wait()
            .map_err(|e| e.to_string())?
            .is_none()
        {
            return Ok(running.info.clone());
        }
    }
    let variant = if variant == "gpu" { "gpu" } else { "cpu" };
    let app2 = app.clone();
    let (child, info) = tauri::async_runtime::spawn_blocking(move || start(&app2, variant))
        .await
        .map_err(|e| e.to_string())??;
    *state.0.lock().unwrap() = Some(Running {
        child,
        info: info.clone(),
    });
    Ok(info)
}

fn start(app: &AppHandle, variant: &str) -> Result<(Child, EngineInfo), String> {
    let project = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("python");
    let root = engine_dir(app)?;
    let env = env_dir(app, variant)?;
    let data = root.parent().ok_or("no data folder")?.to_path_buf();
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;

    // 1. Python, installed by uv into the app's folder only (no PATH shims, no registry entries).
    let uv = uv()?;
    let pythons = root.join("python");
    let cache = root.join("uv-cache");
    let interpreter = match find_python(&pythons) {
        Some(py) => py,
        None => {
            let _ = app.emit("engine-log", format!("Installing Python {PYTHON} with uv…"));
            let installed = run_logged(
                app,
                Command::new(&uv)
                    .args(["python", "install", PYTHON, "--no-bin", "--no-registry"])
                    .env("UV_PYTHON_INSTALL_DIR", &pythons)
                    .env("UV_CACHE_DIR", &cache),
            );
            // uv then links `cpython-3.12-…` to the patch release with a junction/symlink. Where the OS refuses to
            // follow junctions in the user profile (Windows redirection guard) that step fails although Python is
            // installed, so the interpreter's own folder is used and the link is removed.
            find_python(&pythons).ok_or_else(|| {
                format!(
                    "installing Python failed:\n{}",
                    installed.err().unwrap_or_default()
                )
            })?
        }
    };
    remove_links(&pythons);

    // 2. The locked environment. Fast when it is already current; after an app update it installs what changed.
    let _ = app.emit(
        "engine-log",
        format!("Preparing GeoPulse + PyTorch ({variant}) with uv…"),
    );
    run_logged(
        app,
        Command::new(&uv)
            .args([
                "sync", "--frozen",
                "--no-dev", // editable: the engine runs the app's own copy of the package, so updates apply
                "--extra", variant, "--python",
            ])
            .arg(&interpreter)
            .arg("--project")
            .arg(&project)
            .env("UV_PROJECT_ENVIRONMENT", &env)
            .env("UV_CACHE_DIR", &cache)
            .env("UV_PYTHON_DOWNLOADS", "never"),
    )
    .map_err(|e| format!("installing the engine failed:\n{e}"))?;
    let version = app.package_info().version.to_string();
    std::fs::write(root.join(format!("env-{variant}.ok")), version).map_err(|e| e.to_string())?;

    // 3. Trained models from the GitHub release (checksum-verified; files already present are kept).
    let py = python(&env);
    let engine_env = |cmd: &mut Command| {
        cmd.env("GEOPULSE_MODELS", data.join("models"))
            .env("GEOPULSE_OUTPUTS", data.join("outputs"))
            .env("GEOPULSE_CACHE", data.join("cache"))
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1") // never write into the app's (signed, maybe read-only) folder
            .env("PYTHONIOENCODING", "utf-8")
            .current_dir(&data);
    };
    let mut pull = Command::new(&py);
    pull.args(["-m", "geopulse.cli", "models", "pull"]);
    engine_env(&mut pull);
    if let Err(e) = run_logged(app, &mut pull) {
        let _ = app.emit(
            "engine-log",
            format!("! could not update the trained models (offline?): {e}"),
        );
    }

    // 4. The API on a free loopback port, with a per-launch token.
    let port = free_port()?;
    let info = EngineInfo {
        url: format!("http://127.0.0.1:{port}"),
        token: token()?,
    };
    let mut serve = Command::new(&py);
    serve
        .args([
            "-m",
            "geopulse.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
        ])
        .arg(port.to_string())
        .arg("--exit-with-parent") // the engine ends with the app, even if the app crashes
        .arg(std::process::id().to_string())
        .env("GEOPULSE_TOKEN", &info.token)
        .env("GEOPULSE_MAX_JOB_KM2", MAX_JOB_KM2);
    if cfg!(debug_assertions) {
        serve.env("GEOPULSE_CORS_ORIGINS", "http://localhost:5173"); // `tauri dev` serves the UI from Vite
    }
    engine_env(&mut serve);
    let mut child = hidden(&mut serve)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start the engine: {e}"))?;
    let out = forward(app, child.stdout.take());
    let err = forward(app, child.stderr.take());

    let t0 = Instant::now();
    while !healthy(port) {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            let tail = [
                out.join().unwrap_or_default(),
                err.join().unwrap_or_default(),
            ]
            .concat();
            return Err(format!(
                "the engine exited ({status}):\n{}",
                tail.join("\n")
            ));
        }
        if t0.elapsed() > Duration::from_secs(180) {
            let _ = child.kill();
            return Err("the engine did not answer within 3 minutes".into());
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Ok((child, info))
}

fn stop(app: &AppHandle) {
    if let Some(mut running) = app.state::<Engine>().0.lock().unwrap().take() {
        let _ = running.child.kill();
        let _ = running.child.wait();
    }
}

/// Before a self-update: the Windows installer replaces the app files and exits the app without RunEvent::Exit.
#[tauri::command]
fn engine_stop(app: AppHandle) {
    stop(&app);
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Engine::default())
        .invoke_handler(tauri::generate_handler![engine_status, engine_start, engine_stop])
        .build(tauri::generate_context!())
        .expect("error while building GeoPulse")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop(app);
            }
        });
}
