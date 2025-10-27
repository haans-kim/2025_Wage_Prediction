import { app, BrowserWindow, dialog } from 'electron';
import * as path from 'path';
import { spawn } from 'child_process';
import * as fs from 'fs';
import express from 'express';

let mainWindow: any = null;
let backendProcess: any = null;
let frontendServer: any = null;
const isDev = process.env.NODE_ENV === 'development';

// 로그 파일 스트림
let logStream: any = null;

/**
 * 실행 디렉토리 가져오기 (exe 파일이 위치한 디렉토리)
 */
function getAppDataDir(): string {
  if (app.isPackaged) {
    // 패키징된 경우: exe 파일이 있는 디렉토리
    return path.dirname(process.execPath);
  } else {
    // 개발 모드: 프로젝트 루트
    return process.cwd();
  }
}

/**
 * 로그 함수
 */
function log(message: string, ...args: any[]) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message} ${args.map(a => JSON.stringify(a)).join(' ')}\n`;
  console.log(message, ...args);

  // 로그 스트림 초기화
  if (!logStream) {
    try {
      const appDataDir = getAppDataDir();
      const logPath = path.join(appDataDir, 'electron-main.log');
      logStream = fs.createWriteStream(logPath, { flags: 'a' });
      logStream.write(`\n========================================\n`);
      logStream.write(`Electron Main Process Started\n`);
      logStream.write(`Time: ${timestamp}\n`);
      logStream.write(`App Data Dir: ${appDataDir}\n`);
      logStream.write(`========================================\n`);
    } catch (err) {
      console.error('Failed to create log file:', err);
    }
  }

  if (logStream) {
    logStream.write(logMessage);
  }
}

log('=== Electron Main Process Started ===');
log('isDev:', isDev);
log('process.execPath:', process.execPath);

/**
 * 전역 에러 핸들러
 */
process.on('uncaughtException', (error) => {
  log('Uncaught Exception:', error);
  dialog.showErrorBox('Application Error', `Uncaught Exception: ${error.message}\n\nStack: ${error.stack}`);
});

process.on('unhandledRejection', (reason, promise) => {
  log('Unhandled Rejection at:', promise, 'reason:', reason);
});

/**
 * BrowserWindow 생성
 */
function createWindow() {
  log('Creating window...');

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false, // localhost 허용
    },
    title: 'SambioWage',
    show: true,
  });

  // 창이 준비되면 표시
  mainWindow.once('ready-to-show', () => {
    log('Window ready to show');
    mainWindow?.show();
  });

  // 웹 콘텐츠 에러 핸들링
  mainWindow.webContents.on('did-fail-load', (event: any, errorCode: any, errorDescription: any, validatedURL: any) => {
    log('Failed to load:', errorCode, errorDescription, validatedURL);
    dialog.showErrorBox('Page Load Error', `Failed to load page:\nError: ${errorCode}\n${errorDescription}\nURL: ${validatedURL}`);
  });

  mainWindow.webContents.on('did-finish-load', () => {
    log('Page loaded successfully');
  });

  // Backend와 Frontend 시작
  startBackend();
  startFrontend();
  waitForServers();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * Backend Python 프로세스 시작
 */
function startBackend() {
  try {
    log('=== Starting Backend (Python FastAPI) ===');

    const appDataDir = getAppDataDir();

    // Python 실행 파일 경로
    const pythonExe = app.isPackaged
      ? path.join(process.resourcesPath, 'python-runtime', 'python.exe')
      : path.join(appDataDir, 'resources', 'python-runtime', 'python.exe');

    // Backend 스크립트 경로
    const backendScript = app.isPackaged
      ? path.join(process.resourcesPath, 'app', 'backend', 'run.py')
      : path.join(appDataDir, 'backend', 'run.py');

    // Backend 디렉토리
    const backendDir = path.dirname(backendScript);

    log('Python exe:', pythonExe);
    log('Backend script:', backendScript);
    log('Backend dir:', backendDir);
    log('Python exists:', fs.existsSync(pythonExe));
    log('Backend script exists:', fs.existsSync(backendScript));

    if (!fs.existsSync(pythonExe)) {
      const errorMsg = `Python not found at: ${pythonExe}`;
      log(errorMsg);
      dialog.showErrorBox('Python Error', errorMsg);
      return;
    }

    if (!fs.existsSync(backendScript)) {
      const errorMsg = `Backend script not found at: ${backendScript}`;
      log(errorMsg);
      dialog.showErrorBox('Backend Error', errorMsg);
      return;
    }

    // 환경 변수 설정
    const env = {
      ...process.env,
      ELECTRON_APP: 'true',
      APP_DATA_DIR: appDataDir,
      PORT: '8000',
      PYTHONPATH: backendDir,
    };

    log('Starting backend process...');
    log('Environment:', JSON.stringify(env, null, 2));

    // Python 프로세스 시작
    backendProcess = spawn(pythonExe, [backendScript], {
      cwd: backendDir,
      env: env,
      shell: false,
      windowsHide: false,
    });

    if (!backendProcess) {
      log('Failed to spawn backend process');
      return;
    }

    backendProcess.on('error', (error: any) => {
      log('Failed to start backend:', error);
      dialog.showErrorBox('Backend Error', `Failed to start backend.\n\nError: ${error.message}`);
    });

    backendProcess.on('spawn', () => {
      log('Backend process spawned successfully');
    });

    backendProcess.on('exit', (code: any, signal: any) => {
      log(`Backend exited with code ${code} and signal ${signal}`);
      if (code !== 0 && code !== null) {
        dialog.showErrorBox('Backend Crashed',
          `Backend stopped unexpectedly.\n\nExit code: ${code}\nSignal: ${signal}`);
      }
    });

    // 로그 파일
    const backendLogPath = path.join(appDataDir, 'backend.log');
    const backendLogStream = fs.createWriteStream(backendLogPath, { flags: 'a' });
    log('Backend log file:', backendLogPath);

    backendProcess.stdout?.on('data', (data: any) => {
      const msg = data.toString();
      log('[Backend STDOUT]', msg);
      backendLogStream.write(msg);
    });

    backendProcess.stderr?.on('data', (data: any) => {
      const msg = data.toString();
      log('[Backend STDERR]', msg);
      backendLogStream.write(msg);
    });

  } catch (error) {
    log('Error starting backend:', error);
    dialog.showErrorBox('Backend Error', `Error: ${error}`);
  }
}

/**
 * Frontend Express 서버 시작
 */
function startFrontend() {
  try {
    log('=== Starting Frontend (Express) ===');

    const appDataDir = getAppDataDir();
    const buildPath = app.isPackaged
      ? path.join(process.resourcesPath, 'app', 'frontend', 'build')
      : path.join(appDataDir, 'frontend', 'build');

    log('Frontend build path:', buildPath);
    log('Build path exists:', fs.existsSync(buildPath));

    if (!fs.existsSync(buildPath)) {
      const errorMsg = `Frontend build not found at: ${buildPath}`;
      log(errorMsg);
      dialog.showErrorBox('Frontend Error', errorMsg);
      return;
    }

    const expressApp = express();

    // 정적 파일 서빙
    expressApp.use(express.static(buildPath));

    // SPA 라우팅: 모든 요청을 index.html로
    expressApp.get('*', (req: any, res: any) => {
      res.sendFile(path.join(buildPath, 'index.html'));
    });

    // 서버 시작
    frontendServer = expressApp.listen(3000, () => {
      log('Frontend server started on http://localhost:3000');
    });

    frontendServer.on('error', (error: any) => {
      log('Frontend server error:', error);
      dialog.showErrorBox('Frontend Error', `Failed to start frontend server.\n\nError: ${error.message}`);
    });

  } catch (error) {
    log('Error starting frontend:', error);
    dialog.showErrorBox('Frontend Error', `Error: ${error}`);
  }
}

/**
 * 서버 준비 대기
 */
async function waitForServers() {
  log('Loading application...');

  // 서버 시작까지 짧은 대기 (3초)
  const waitTime = 3000;
  log(`Waiting ${waitTime}ms for servers to initialize...`);
  await new Promise(resolve => setTimeout(resolve, waitTime));

  // BrowserWindow에 로드
  if (mainWindow && !mainWindow.isDestroyed()) {
    log('Loading URL: http://localhost:3000');
    try {
      await mainWindow.loadURL('http://localhost:3000');
      log('✓ URL loaded successfully');
      return;
    } catch (error: any) {
      log('Error loading URL:', error.message);
      log('Error stack:', error.stack);
      // 로드 실패 시 재시도
      log('Retrying URL load in 5 seconds...');
      await new Promise(resolve => setTimeout(resolve, 5000));
      try {
        await mainWindow.loadURL('http://localhost:3000');
        log('✓ URL loaded successfully on retry');
        return;
      } catch (retryError: any) {
        log('Retry failed:', retryError.message);
        log('Retry error stack:', retryError.stack);
      }
    }
  } else {
    log('MainWindow is null or destroyed');
  }

  // 로드 실패
  log('✗ Failed to load application');
  dialog.showErrorBox('Application Error',
    'Failed to load application.\n\n' +
    'Please check if the backend server is running on port 8000.');
}

/**
 * 앱 종료 시 프로세스 정리
 */
function cleanup() {
  log('Cleaning up processes...');

  if (backendProcess) {
    log('Killing backend process...');
    backendProcess.kill();
    backendProcess = null;
  }

  if (frontendServer) {
    log('Closing frontend server...');
    frontendServer.close();
    frontendServer = null;
  }

  if (logStream) {
    logStream.end();
    logStream = null;
  }

  log('Cleanup complete');
}

/**
 * 앱 이벤트 핸들러
 */
app.whenReady().then(() => {
  log('App ready, creating window...');
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  log('All windows closed');
  cleanup();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  log('Before quit');
  cleanup();
});

app.on('will-quit', () => {
  log('Will quit');
});

app.on('quit', () => {
  log('App quit');
});
