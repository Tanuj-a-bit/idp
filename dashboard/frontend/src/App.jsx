import React, { useState, useEffect, useCallback } from 'react';
import DeckGL from '@deck.gl/react';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { ScatterplotLayer, IconLayer, PathLayer } from '@deck.gl/layers';
import { Ship, Radio, Gauge, MapPin, Activity, Navigation } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Mock target if websocket not connected
const INITIAL_VIEW_STATE = {
  longitude: 103.851959,
  latitude: 1.290270,
  zoom: 3, // Zoomed out to show the globe nicely
  pitch: 35,
  bearing: 0
};

const MAPBOX_ACCESS_TOKEN = ''; // Removed for security. Use environment variables.

export default function App() {
  const [vessels, setVessels] = useState({});
  const [selectedMmsi, setSelectedMmsi] = useState(null);
  const [wsStatus, setWsStatus] = useState('connecting');
  
  const bufferRef = React.useRef({});

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/tracks');
    ws.onopen = () => setWsStatus('connected');
    ws.onclose = () => setWsStatus('disconnected');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      bufferRef.current[data.mmsi] = data;
    };

    const flushInterval = setInterval(() => {
        if (Object.keys(bufferRef.current).length === 0) return;
        
        setVessels(prev => {
            const next = { ...prev };
            Object.keys(bufferRef.current).forEach(mmsi => {
                const data = bufferRef.current[mmsi];
                next[mmsi] = {
                    ...data,
                    history: [...(prev[mmsi]?.history || []).slice(-20), [data.lon, data.lat]]
                };
            });
            return next;
        });
        bufferRef.current = {};
    }, 500);

    return () => {
        ws.close();
        clearInterval(flushInterval);
    };
  }, []);

  const layers = React.useMemo(() => [
    new IconLayer({
      id: 'vessel-icons',
      data: Object.values(vessels),
      pickable: true,
      iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
      iconMapping: {
        marker: { x: 0, y: 0, width: 128, height: 128, mask: true }
      },
      getIcon: d => 'marker',
      getPosition: d => [d.lon, d.lat],
      getSize: 40,
      getColor: d => d.mmsi === selectedMmsi ? [248, 113, 113] : [56, 189, 248],
      onClick: ({object}) => setSelectedMmsi(object.mmsi),
      updateTriggers: {
        getColor: [selectedMmsi]
      }
    }),
  ], [vessels, selectedMmsi]);

  const selectedVessel = vessels[selectedMmsi];

  return (
    <div className="relative w-screen h-screen">
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >
        <Map
          mapStyle="https://tiles.basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        />
      </DeckGL>

      {/* Sidebar Overlay */}
      <div className="absolute top-4 left-4 z-10 w-80 flex flex-col gap-4">
        <header className="bg-slate-900/80 backdrop-blur-md p-4 rounded-xl border border-slate-700/50 shadow-2xl">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${wsStatus === 'connected' ? 'bg-emerald-400' : 'bg-rose-400'} animate-pulse`} />
            <h1 className="text-lg font-bold tracking-tight text-slate-100 uppercase">Aegis Maritime</h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">Real-time AI Tracking & Prediction</p>
        </header>

        <section className="bg-slate-900/80 backdrop-blur-md p-4 rounded-xl border border-slate-700/50 shadow-2xl overflow-y-auto max-h-[70vh]">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Radio size={14} /> Active Vessels ({Object.keys(vessels).length})
          </h2>
          
          <div className="flex flex-col gap-2">
            {Object.values(vessels).map((v) => (
              <button 
                key={v.mmsi}
                onClick={() => setSelectedMmsi(v.mmsi)}
                className={`flex items-center gap-3 p-3 rounded-lg transition-all text-left ${selectedMmsi === v.mmsi ? 'bg-ship-primary/20 border-ship-primary/40' : 'hover:bg-slate-800'}`}
              >
                <div className="p-2 bg-slate-800 rounded-md">
                   <Ship size={18} className="text-ship-primary" />
                </div>
                <div>
                  <div className="text-sm font-medium">{v.name || 'Unknown Vessel'}</div>
                  <div className="text-[10px] text-slate-500 font-mono">{v.mmsi}</div>
                </div>
              </button>
            ))}
            {Object.keys(vessels).length === 0 && (
              <div className="py-8 text-center text-slate-500 italic text-sm">
                No active signals detected...
              </div>
            )}
          </div>
        </section>
      </div>

      <AnimatePresence>
        {selectedVessel && (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="absolute top-4 right-4 z-10 w-96 bg-slate-900/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl overflow-hidden p-6"
          >
            <div className="flex justify-between items-start mb-6">
              <div>
                <h3 className="text-xl font-bold text-white leading-tight">{selectedVessel.name || 'Class A Vessel'}</h3>
                <span className="bg-ship-primary/10 text-ship-primary px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide border border-ship-primary/30">MMSI: {selectedVessel.mmsi}</span>
              </div>
              <button onClick={() => setSelectedMmsi(null)} className="text-slate-500 hover:text-white transition-colors">✕</button>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
               <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30">
                 <div className="text-[10px] uppercase text-slate-500 mb-1 flex items-center gap-1 font-bold">
                    <Navigation size={10} /> Latitude
                 </div>
                 <div className="text-sm font-mono font-semibold">{selectedVessel.lat.toFixed(4)}°</div>
               </div>
               <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30">
                 <div className="text-[10px] uppercase text-slate-500 mb-1 flex items-center gap-1 font-bold">
                    <Navigation size={10} className="rotate-90" /> Longitude
                 </div>
                 <div className="text-sm font-mono font-semibold">{selectedVessel.lon.toFixed(4)}°</div>
               </div>
               <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30 grid grid-cols-2 col-span-2">
                 <div className="flex flex-col">
                    <div className="text-[10px] uppercase text-slate-500 mb-1 flex items-center gap-1 font-bold">
                        <Gauge size={10} /> Speed (SOG)
                    </div>
                    <div className="text-sm font-mono font-semibold">12.4 kn</div>
                 </div>
                 <div className="flex flex-col border-l border-slate-700/50 pl-3">
                    <div className="text-[10px] uppercase text-slate-500 mb-1 flex items-center gap-1 font-bold">
                        <Activity size={10} /> Status
                    </div>
                    <div className="text-sm font-bold text-emerald-400">Navigating</div>
                 </div>
               </div>
            </div>
            
            {/* Real-time YOLO AI Surveillance Feed */}
            <div className="mb-6 rounded-xl overflow-hidden border border-slate-700/50 relative bg-black aspect-video flex-shrink-0">
               <div className="absolute top-2 left-2 z-20 flex gap-2">
                  <span className="bg-rose-500/80 text-white text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider backdrop-blur-sm animate-pulse flex items-center gap-1">
                     <div className="w-1.5 h-1.5 bg-white rounded-full"></div> AI INTERCEPT
                  </span>
                  <span className="bg-slate-900/80 text-white text-[9px] font-mono px-1.5 py-0.5 rounded backdrop-blur-sm border border-slate-700">
                     CAM-YOLO-V8
                  </span>
               </div>
               
               {selectedVessel.frame ? (
                  <img 
                    src={`data:image/jpeg;base64,${selectedVessel.frame}`}
                    alt="AI Detection Feed"
                    className="w-full h-full object-contain bg-black"
                  />
               ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-slate-600 gap-2">
                     <div className="w-8 h-8 border-2 border-slate-700 border-t-ship-primary rounded-full animate-spin" />
                     <span className="text-[10px] font-mono uppercase tracking-widest">Awaiting Signal...</span>
                  </div>
               )}
               
               {/* Tactical Scanline Overlay */}
               <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] bg-[length:100%_4px,3px_100%] z-30 opacity-20" />
            </div>

            <div className="border-t border-slate-700/50 pt-6">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <MapPin size={14} className="text-ship-secondary" /> Predicted Trajectory (30m)
              </h4>
              <div className="h-24 bg-slate-800/50 rounded-xl border border-ship-secondary/20 flex flex-col items-center justify-center text-center p-4">
                 <p className="text-xs text-emerald-400 font-mono font-bold mb-2 break-all whitespace-normal px-2">AIS & LSTM Trajectory Active</p>
                 <div className="text-[10px] text-slate-400 font-mono">
                   [{selectedVessel.prediction && selectedVessel.prediction.length > 0 ? selectedVessel.prediction.map(p => `${p[0].toFixed(3)},${p[1].toFixed(3)}`).slice(0,2).join(' → ') : 'Calculating...'}]
                 </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
