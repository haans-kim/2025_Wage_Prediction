/**
 * Preload script for SambioWage Electron App
 *
 * This script runs before the renderer process loads.
 * It provides a secure bridge between the renderer and main process.
 */

// Currently, we don't need to expose any APIs to the renderer process
// since the React app communicates directly with the backend via HTTP

console.log('Preload script loaded');

// Export empty object to satisfy TypeScript
export {};
