"""AirSplatMap Web Dashboard - Real-time 3D Gaussian Splatting visualization"""
import asyncio,json,time,threading,queue,logging,sys,base64,io,pickle
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).parent.parent))
logger=logging.getLogger(__name__)
try:
    from websockets.server import serve as ws_serve
except:ws_serve=None
import http.server,socketserver
from urllib.parse import urlparse
HIST=Path(__file__).parent.parent/"output"/".history"
HIST.mkdir(parents=True,exist_ok=True)

class Server:
    def __init__(s,hp=9002,wp=9003):
        s.hp,s.wp=hp,wp
        s._cl,s._run=set(),False
        s._st,s._eng,s._th,s._stop={},{},{},{}
        s._q=queue.Queue()
        s._ds=s._scan()
        s._lk=threading.Lock()
        s._hist=s._loadh()
        s._snaps={}
        s._cur_key=None
        s._settings={'target_fps':30,'unlimited':False,'pts_3d':20000,'splat_size':2.0,'splat_opacity':0.7,'render_engine':'auto','snap_pts':50000}

    def _scan(s):
        ds=[]
        base=Path(__file__).parent.parent.parent/"datasets"
        for cat,p in[("TUM",base/"tum"),("T&T",base/"tandt")]:
            if not p.exists():continue
            for d in sorted(p.iterdir()):
                if not d.is_dir():continue
                fr=sum(1 for l in open(d/"rgb.txt")if l.strip()and not l.startswith('#'))if(d/"rgb.txt").exists()else 0
                ds.append({'name':d.name,'path':str(d),'frames':fr,'type':cat})
        return ds

    def _loadh(s):
        h=[]
        for f in sorted(HIST.glob("*.json"),reverse=True)[:50]:
            try:h.append(json.load(open(f)))
            except:pass
        return h

    def start(s):
        s._run=True
        threading.Thread(target=s._http,daemon=True).start()
        if ws_serve:threading.Thread(target=s._ws,daemon=True).start()

    def stop(s):
        s._run=False
        for k in s._stop:s._stop[k].set()

    def _http(s):
        with socketserver.TCPServer(("",s.hp),s._handler(),bind_and_activate=True)as srv:
            srv.socket.setsockopt(1,2,1)
            while s._run:srv.handle_request()

    def _ws(s):
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(s._wsl())

    async def _wsl(s):
        async with ws_serve(s._wsh,"0.0.0.0",s.wp):
            while s._run:
                while not s._q.empty():
                    try:await s._bc(s._q.get_nowait())
                    except:break
                await asyncio.sleep(0.025)

    async def _wsh(s,ws):
        s._cl.add(ws)
        try:
            await ws.send(json.dumps({'type':'init','engines':s._engines(),'datasets':s._ds,'status':s._st,'history':s._hist,'current':s._cur_key,'settings':s._settings}))
            async for m in ws:
                d=json.loads(m)
                cmd=d.get('cmd')
                if cmd=='start':s._start(d['engine'],d['dataset'])
                elif cmd=='stop':
                    key=d.get('key')
                    if key:s._stp(key)
                elif cmd=='stop_all':
                    for k in list(s._stop):s._stop[k].set()
                    s._q.put({'type':'stopped'})
                elif cmd=='clear':
                    with s._lk:s._st.clear()
                    s._cur_key=None
                    s._q.put({'type':'cleared'})
                elif cmd=='settings':
                    s._settings.update(d.get('data',{}))
                    s._q.put({'type':'settings','data':s._settings})
                elif cmd=='replay':await s._replay(ws,d.get('key'),d.get('frame',0))
                elif cmd=='replay_range':await s._replay_range(ws,d.get('key'),d.get('end_frame',0))
                elif cmd=='get_snaps':await s._snapi(ws,d.get('key'))
        except:pass
        finally:s._cl.discard(ws)

    async def _replay(s,ws,key,fr):
        if key not in s._snaps:
            sf=HIST/f"{key}_snaps.pkl"
            if sf.exists():
                try:s._snaps[key]=pickle.load(open(sf,'rb'))
                except:pass
        if key in s._snaps and 0<=fr<len(s._snaps[key]):
            sn=s._snaps[key][fr]
            await ws.send(json.dumps({'type':'replay_frame','key':key,'frame':fr,'total':len(s._snaps[key]),**sn}))

    async def _replay_range(s,ws,key,end_fr):
        if key not in s._snaps:
            sf=HIST/f"{key}_snaps.pkl"
            if sf.exists():
                try:s._snaps[key]=pickle.load(open(sf,'rb'))
                except:pass
        if key in s._snaps:
            metrics=[]
            for i in range(min(end_fr+1,len(s._snaps[key]))):
                sn=s._snaps[key][i]
                metrics.append({'fps':sn.get('fps',0),'loss':sn.get('loss',0),'gaussians':sn.get('gaussians',0),'psnr':sn.get('psnr',0)})
            await ws.send(json.dumps({'type':'replay_metrics','key':key,'metrics':metrics,'frame':end_fr}))

    async def _snapi(s,ws,key):
        n=len(s._snaps.get(key,[]))
        if n==0:
            sf=HIST/f"{key}_snaps.pkl"
            if sf.exists():
                try:s._snaps[key]=pickle.load(open(sf,'rb'));n=len(s._snaps[key])
                except:pass
        await ws.send(json.dumps({'type':'snap_info','key':key,'total':n}))

    async def _bc(s,m):
        for ws in list(s._cl):
            try:await ws.send(json.dumps(m))
            except:s._cl.discard(ws)

    def _engines(s):
        from src.engines import list_engines
        return[{'name':n,**v}for n,v in list_engines().items()]

    def _start(s,eng,ds):
        key=f"{eng}_{ds}"
        if key in s._th and s._th[key].is_alive():return
        s._stop[key]=threading.Event()
        s._snaps[key]=[]
        s._cur_key=key
        with s._lk:s._st[key]={'name':eng,'dataset':ds,'status':'starting','frame':0,'total':0,'fps':0,'loss':0,'gaussians':0,'elapsed':0}
        t=threading.Thread(target=s._run_eng,args=(eng,ds,key),daemon=True)
        s._th[key]=t;t.start()
        s._q.put({'type':'current','key':key,'engine':eng,'dataset':ds})

    def _stp(s,key):
        if key in s._stop:
            s._stop[key].set()
            s._q.put({'type':'stopped','key':key})

    def _safe_colors(s,engine,idx=None):
        try:
            if hasattr(engine,'get_gaussian_colors'):
                cols=engine.get_gaussian_colors()
                if cols is not None and len(cols)>0:
                    cols=np.asarray(cols,dtype=np.float32)
                    if cols.max()>1:cols=cols/255.0
                    if idx is not None:cols=cols[idx]
                    return base64.b64encode(cols.astype(np.float32).tobytes()).decode()
        except:pass
        return ""

    def _safe_pc(s,engine,max_pts):
        try:
            pc=engine.get_point_cloud()
            if pc is None:return None,None
            pc=np.asarray(pc)
            if len(pc)==0:return None,None
            idx=None
            if len(pc)>max_pts:
                idx=np.random.choice(len(pc),max_pts,replace=False)
                pc=pc[idx]
            return pc,idx
        except:return None,None

    def _sw_render(s,engine,pose,size,intr):
        try:
            pc=engine.get_point_cloud()
            cols=engine.get_gaussian_colors()if hasattr(engine,'get_gaussian_colors')else None
            if pc is None or len(pc)==0:
                return ""
            pc=np.asarray(pc)
            pose=np.asarray(pose)
            # Pose is camera-to-world (T_wc), we need world-to-camera (T_cw)
            R=pose[:3,:3]
            t=pose[:3,3]
            cam_pts=(pc-t)@R
            
            # Handle both dict and array intrinsics
            if isinstance(intr,dict):
                fx,fy,cx,cy=intr.get('fx',525),intr.get('fy',525),intr.get('cx',319.5),intr.get('cy',239.5)
                orig_w,orig_h=intr.get('width',640),intr.get('height',480)
            else:
                fx,fy,cx,cy=intr[0],intr[1],intr[2],intr[3]
                orig_w,orig_h=640,480
            
            h,w=size
            scale_x=w/orig_w
            scale_y=h/orig_h
            
            valid=(cam_pts[:,2]>0.1)
            if not np.any(valid):
                return ""
            
            z=cam_pts[:,2]
            u_orig=cam_pts[:,0]*fx/z+cx
            v_orig=cam_pts[:,1]*fy/z+cy
            u=(u_orig*scale_x).astype(int)
            v=(v_orig*scale_y).astype(int)
            
            valid&=(u>=0)&(u<w)&(v>=0)&(v<h)
            img=np.zeros((h,w,3),dtype=np.uint8)
            uu,vv=u[valid],v[valid]
            zz=z[valid]
            if len(uu)==0:
                return ""
            
            # Sort by depth (far to near)
            order=np.argsort(-zz)
            uu,vv,zz=uu[order],vv[order],zz[order]
            
            # Get colors
            if cols is not None and len(cols)>=len(pc):
                c=np.asarray(cols)[valid][order]
                if c.max()<=1:c=(c*255).astype(np.uint8)
                if len(c.shape)==1 or c.shape[1]<3:
                    c=np.tile(c.reshape(-1,1),(1,3))
                c=c[:,:3]
            else:
                c=np.full((len(uu),3),[100,180,255],dtype=np.uint8)
            
            # Draw splat-like circles based on depth (closer = bigger)
            # Use OpenCV for efficient circle drawing if available, else manual
            try:
                import cv2
                for i in range(len(uu)):
                    # Splat radius inversely proportional to depth
                    radius=max(1,int(3.0/zz[i]))  # Closer points get bigger radius
                    radius=min(radius,5)  # Cap at 5 pixels
                    cv2.circle(img,(uu[i],vv[i]),radius,c[i].tolist(),-1)
            except ImportError:
                # Fallback: draw 3x3 blocks for each point
                for i in range(len(uu)):
                    x,y=uu[i],vv[i]
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            nx,ny=x+dx,y+dy
                            if 0<=nx<w and 0<=ny<h:
                                img[ny,nx]=c[i]
            
            from PIL import Image
            buf=io.BytesIO();Image.fromarray(img).save(buf,format='JPEG',quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.error(f"SW render error: {e}")
            return ""

    def _render(s,engine,pose,size,intr):
        """Render using configured method.
        
        Render modes:
        - 'auto': Try native engine render, fall back to software
        - 'sw': Force software point-cloud rendering
        - 'native': Force native engine render (error if unavailable)
        """
        mode=s._settings.get('render_engine','auto')
        
        # Software-only mode
        if mode == 'sw':
            return s._sw_render(engine,pose,size,intr)
        
        # Try native render_view (auto or native mode)
        if hasattr(engine,'render_view'):
            try:
                # render_view expects (pose, (width, height)) and returns HxWx3 numpy array
                rendered=engine.render_view(pose,(size[1],size[0]))  # size is (h,w), render_view wants (w,h)
                if rendered is not None and len(rendered) > 0:
                    from PIL import Image
                    if hasattr(rendered,'cpu'):rendered=rendered.cpu().numpy()
                    rendered=np.asarray(rendered)
                    if rendered.max()<=1:rendered=(rendered*255).astype(np.uint8)
                    if len(rendered.shape)==2:rendered=np.stack([rendered]*3,axis=-1)
                    buf=io.BytesIO();Image.fromarray(rendered).save(buf,format='JPEG',quality=75)
                    return base64.b64encode(buf.getvalue()).decode()
            except Exception as e:
                logger.debug(f"Native render_view failed: {e}")
                if mode == 'native':
                    return ""  # Don't fall back if native was explicitly requested
        
        # Fall back to software render (auto mode)
        return s._sw_render(engine,pose,size,intr)

    def _run_eng(s,eng,ds,key):
        try:
            from src.engines import get_engine
            from src.pipeline.frames import TumRGBDSource
            
            di=next((d for d in s._ds if d['name']==ds),None)
            if not di:raise ValueError(f"Dataset {ds} not found")
            engine=get_engine(eng)
            s._eng[key]=engine
            src=TumRGBDSource(di['path'])
            frames=list(src)
            total=len(frames)
            if total==0:raise ValueError("Empty")
            intr=frames[0].intrinsics
            # Get actual image dimensions for proper aspect ratio
            img_h,img_w=frames[0].rgb.shape[:2]
            # Calculate thumbnail size maintaining aspect ratio
            thumb_w=400
            thumb_h=int(thumb_w*img_h/img_w)
            snap_w=320
            snap_h=int(snap_w*img_h/img_w)
            with s._lk:s._st[key]['total']=total;s._st[key]['status']='running'
            engine.initialize_scene(intr,{'num_frames':total})
            t0=time.time()
            lb=0
            frame_time=1.0/s._settings['target_fps'] if not s._settings['unlimited'] else 0
            
            for i,fr in enumerate(frames):
                ft_start=time.time()
                if s._stop[key].is_set():
                    with s._lk:s._st[key]['status']='stopped'
                    break
                try:engine.add_frame(fr.idx,fr.rgb,fr.depth,fr.pose)
                except:continue
                mt={}
                try:mt=engine.optimize_step(n_steps=5)
                except TypeError:
                    try:mt=engine.optimize_step(num_steps=5)
                    except:
                        try:mt=engine.optimize_step()
                        except:pass
                except:pass
                
                el=time.time()-t0
                fps=(i+1)/el if el>0 else 0
                ng=engine.get_num_gaussians()
                loss=mt.get('loss',0)if isinstance(mt,dict)else 0
                psnr=mt.get('psnr',0)if isinstance(mt,dict)else 0
                
                # Snapshot for replay
                try:
                    pc,idx=s._safe_pc(engine,s._settings['snap_pts'])
                    pb=base64.b64encode(pc.astype(np.float32).tobytes()).decode()if pc is not None else""
                    cb=s._safe_colors(engine,idx)
                    ib=""
                    try:
                        from PIL import Image
                        im=Image.fromarray(fr.rgb);im.thumbnail((snap_w,snap_h))
                        buf=io.BytesIO();im.save(buf,format='JPEG',quality=60)
                        ib=base64.b64encode(buf.getvalue()).decode()
                    except:pass
                    rb=s._sw_render(engine,fr.pose,(snap_h,snap_w),intr)
                    sn={'points':pb,'colors':cb,'image':ib,'rendered':rb,'fps':round(fps,2),'loss':round(loss,4),'gaussians':ng,'psnr':round(psnr,2),'frame_idx':i,'elapsed':round(el,1)}
                    s._snaps[key].append(sn)
                except:pass
                
                # Live update at 5Hz to reduce lag
                if time.time()-lb>=0.2:
                    lb=time.time()
                    max_pts=min(s._settings.get('pts_3d',20000),10000)  # Cap at 10k for performance
                    pc,idx=s._safe_pc(engine,max_pts)
                    pb=base64.b64encode(pc.astype(np.float32).tobytes()).decode()if pc is not None else""
                    cb=s._safe_colors(engine,idx)
                    ib=""
                    try:
                        from PIL import Image
                        im=Image.fromarray(fr.rgb);im.thumbnail((thumb_w,thumb_h))
                        buf=io.BytesIO();im.save(buf,format='JPEG',quality=75)
                        ib=base64.b64encode(buf.getvalue()).decode()
                    except:pass
                    rb=s._render(engine,fr.pose,(thumb_h,thumb_w),intr)
                    with s._lk:s._st[key].update({'frame':i+1,'fps':round(fps,2),'loss':round(loss,4),'gaussians':ng,'elapsed':round(el,1),'psnr':round(psnr,2)})
                    s._q.put({'type':'update','key':key,'frame':i+1,'total':total,'fps':round(fps,2),'loss':round(loss,4),'gaussians':ng,'elapsed':round(el,1),'psnr':round(psnr,2),'points':pb,'colors':cb,'image':ib,'rendered':rb})
                
                if frame_time>0:
                    elapsed=time.time()-ft_start
                    if elapsed<frame_time:time.sleep(frame_time-elapsed)
            
            # Refine
            if not s._stop[key].is_set():
                with s._lk:s._st[key]['status']='refining'
                for _ in range(50):
                    if s._stop[key].is_set():break
                    try:engine.optimize_step(n_steps=1)
                    except:
                        try:engine.optimize_step(num_steps=1)
                        except:
                            try:engine.optimize_step()
                            except:break
            
            ft=time.time()-t0
            fg=engine.get_num_gaussians()
            status='stopped'if s._stop[key].is_set()else'complete'
            with s._lk:s._st[key].update({'status':status,'elapsed':round(ft,1),'gaussians':fg})
            
            run={'key':key,'engine':eng,'dataset':ds,'timestamp':datetime.now().isoformat(),'total_frames':total,'final_gaussians':fg,'avg_fps':round(total/ft,2)if ft>0 else 0,'elapsed_sec':round(ft,1),'status':status}
            s._hist.insert(0,run)
            try:
                with open(HIST/f"{key}_{datetime.now().strftime('%H%M%S')}.json",'w')as f:json.dump(run,f)
            except:pass
            if s._snaps.get(key):
                try:
                    with open(HIST/f"{key}_snaps.pkl",'wb')as f:pickle.dump(s._snaps[key],f)
                except:pass
            s._q.put({'type':'complete','key':key,'run':run,'snap_count':len(s._snaps.get(key,[]))})
        except Exception as e:
            logger.error(f"Engine error: {e}")
            with s._lk:s._st[key]['status']='error';s._st[key]['error']=str(e)
            s._q.put({'type':'error','key':key,'error':str(e)})

    def _handler(s):
        srv=s
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                p=urlparse(self.path).path
                if p in('/','/index.html'):self._r(200,'text/html',HTML)
                elif p=='/api/engines':self._j(srv._engines())
                elif p=='/api/datasets':self._j(srv._ds)
                elif p=='/api/status':self._j(srv._st)
                elif p=='/api/history':self._j(srv._hist)
                elif p=='/api/settings':self._j(srv._settings)
                else:self.send_error(404)
            def do_POST(self):
                p=urlparse(self.path).path
                b=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))).decode()or'{}')
                if p=='/api/start':srv._start(b['engine'],b['dataset']);self._j({'ok':1})
                elif p=='/api/stop':srv._stp(b.get('key'));self._j({'ok':1})
                elif p=='/api/settings':srv._settings.update(b);self._j(srv._settings)
                else:self.send_error(404)
            def _j(self,d):self._r(200,'application/json',json.dumps(d))
            def _r(self,c,t,d):
                self.send_response(c);self.send_header('Content-Type',t);self.send_header('Access-Control-Allow-Origin','*');self.end_headers()
                self.wfile.write(d.encode()if isinstance(d,str)else d)
            def log_message(self,*a):pass
        return H

HTML_PATH=Path(__file__).parent/"web_dashboard.html"
HTML=""

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--http-port",type=int,default=9002)
    p.add_argument("--ws-port",type=int,default=9003)
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    global HTML
    HTML=HTML_PATH.read_text()if HTML_PATH.exists()else"<h1>HTML not found</h1>"
    HTML=HTML.replace('WS_PORT',str(a.ws_port))
    s=Server(a.http_port,a.ws_port)
    s.start()
    print(f"\n  AirSplatMap Dashboard: http://localhost:{a.http_port}\n")
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:
        s.stop()

if __name__=="__main__":main()
