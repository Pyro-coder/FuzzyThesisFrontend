// desktop/main.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { execFile } = require('child_process');

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadURL('http://127.0.0.1:5000');
    mainWindow.on('closed', function () {
        app.quit();
    });
}

app.on('ready', () => {
    // Start the Flask server using the bundled executable
    const flaskExecutable = path.join(__dirname, '..', 'pclr_exam', 'dist', 'app');
    const flaskProcess = execFile(flaskExecutable, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error}`);
            return;
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
    });

    createWindow();
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
