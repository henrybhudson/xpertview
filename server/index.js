import http from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import twilio from "twilio";

const tw = twilio();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const clientDir = path.join(__dirname, '..', 'client');
app.use(express.static(clientDir));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

const rooms = new Map();
function joinRoom(ws, roomId) {
        if (!rooms.has(roomId)) rooms.set(roomId, new Set());
        rooms.get(roomId).add(ws);
        ws._roomId = roomId;
}
function leaveRoom(ws) {
        const room = rooms.get(ws._roomId);
        if (room) { room.delete(ws); if (room.size === 0) rooms.delete(ws._roomId); }
}

app.get("/ice", async (_req, res) => {
        try {
                const token = await tw.tokens.create();
                res.json({ iceServers: token.iceServers });
        } catch (e) {
                console.error(e);
                res.status(500).json({ error: "ICE server fetch failed" });
        }
});

wss.on('connection', (ws) => {
        console.log('[Server] A client connected.');
        ws.on('message', (raw) => {
                let msg; try { msg = JSON.parse(raw); } catch { return; }
                const { type, roomId, payload } = msg;

                console.log(`[Server] Received message: ${type}, ${roomId}`);

                if (type === 'create') {
                        console.log(`[Server] Creating room: ${roomId}`);
                        const already = rooms.get(roomId)?.size || 0; // how many were waiting before creating

                        joinRoom(ws, roomId);
                        ws.send(JSON.stringify({ type: 'created', roomId }));

                        // notify everyone else that a peer is present (so learner gets notified too)
                        rooms.get(roomId)?.forEach(peer => {
                                if (peer !== ws) peer.send(JSON.stringify({ type: 'peer-joined' }));
                        });

                        // if someone was already in the room (learner-first case), notify the creator (coach)
                        if (already > 0) {
                                ws.send(JSON.stringify({ type: 'peer-joined' }));
                        }
                        return;
                }
                if (type === 'join') {
                        console.log(`[Server] Joining room: ${roomId}`);
                        joinRoom(ws, roomId);
                        ws.send(JSON.stringify({ type: 'joined', roomId }));
                        rooms.get(roomId)?.forEach(peer => {
                                if (peer !== ws) peer.send(JSON.stringify({ type: 'peer-joined' }));
                        });
                        return;
                }
                rooms.get(ws._roomId)?.forEach(peer => {
                        if (peer !== ws) peer.send(JSON.stringify({ type, payload }));
                });
        });
        ws.on('close', () => {
                console.log('[Server] A client disconnected.');
                leaveRoom(ws)
        });
});

const PORT = process.env.PORT || 5174;
server.listen(PORT, () => console.log(`Signalling on http://localhost:${PORT}`));
