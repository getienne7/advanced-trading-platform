"""
Analytics Service - Real-time P&L attribution and performance analytics
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from analytics_engine import AnalyticsEngine
from performance_calculator import PerformanceCalculator
from pnl_attribution import PnLAttributionEngine
from risk_metrics_calculator import RiskMetricsCalculator
from websocket_manager import WebSocketManager
from visualization_engine import VisualizationEngine
from dashboard_widgets import DashboardWidgets
from report_generator import ReportGenerator
from notification_service import NotificationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
analytics_engine: Optional[AnalyticsEngine] = None
websocket_manager: Optional[WebSocketManager] = None
visualization_engine: Optional[VisualizationEngine] = None
dashboard_widgets: Optional[DashboardWidgets] = None
report_generator: Optional[ReportGenerator] = None
notification_service: Optional[NotificationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analytics_engine, websocket_manager, visualization_engine, dashboard_widgets, report_generator, notification_service
    
    # Startup
    logger.info("Starting Analytics Service...")
    
    # Initialize components
    analytics_engine = AnalyticsEngine()
    websocket_manager = WebSocketManager()
    visualization_engine = VisualizationEngine()
    dashboard_widgets = DashboardWidgets()
    report_generator = ReportGenerator()
    notification_service = NotificationService()
    
    # Start background tasks
    asyncio.create_task(analytics_engine.start_real_time_processing())
    asyncio.create_task(websocket_manager.start_broadcast_loop())
    
    logger.info("Analytics Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Analytics Service...")
    if analytics_engine:
        await analytics_engine.stop()
    if websocket_manager:
        await websocket_manager.stop()


# Create FastAPI app
app = FastAPI(
    title="Analytics Service",
    description="Real-time P&L attribution and performance analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/analytics/pnl/{user_id}")
async def get_pnl_attribution(
    user_id: str,
    period_hours: int = 24,
    strategy_id: Optional[str] = None
):
    """Get P&L attribution analysis"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not initialized")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=period_hours)
        
        pnl_data = await analytics_engine.get_pnl_attribution(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            strategy_id=strategy_id
        )
        
        return {
            "success": True,
            "data": pnl_data,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": period_hours
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting P&L attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/performance/{user_id}")
async def get_performance_metrics(
    user_id: str,
    period_days: int = 30,
    strategy_id: Optional[str] = None
):
    """Get comprehensive performance metrics"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not initialized")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=period_days)
        
        performance_data = await analytics_engine.get_performance_metrics(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            strategy_id=strategy_id
        )
        
        return {
            "success": True,
            "data": performance_data,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": period_days
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/risk-metrics/{user_id}")
async def get_risk_metrics(user_id: str):
    """Get real-time risk metrics"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not initialized")
        
        risk_metrics = await analytics_engine.get_real_time_risk_metrics(user_id)
        
        return {
            "success": True,
            "data": risk_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/attribution/{user_id}")
async def get_performance_attribution(
    user_id: str,
    period_days: int = 30
):
    """Get detailed performance attribution analysis"""
    try:
        if not analytics_engine:
            raise HTTPException(status_code=503, detail="Analytics engine not initialized")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=period_days)
        
        attribution_data = await analytics_engine.get_performance_attribution(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "success": True,
            "data": attribution_data,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": period_days
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/charts/pnl/{user_id}")
async def get_pnl_chart(
    user_id: str,
    timeframe: str = "1D"
):
    """Get P&L attribution chart"""
    try:
        if not analytics_engine or not visualization_engine:
    ed")
        
        # Get P&L data
        end_time = datetime.utcnow()
        4)
        
        pnl_data = tion(
            userser_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create chart
        chart = frame)
        
        return {
            
            "data": chart
        }
        
:
}")
        raise HTTPExceptio


@app.get("/analytics/chr_id}")
async def get_perf):
    """Get performant"""
    try:
     _id)client(usere_anager.removcket_m await webso          er:
 ocket_managf webs      i  ager
ent from man# Remove cli        lly:
)
    finaid}: {e}" user {user_error forebSocket error(f"W     logger.n as e:
   ceptioxcept Ex 
    e          ak
         bre         onnect:
   etDisccept WebSock     exge)
       er_id, messassage(usmeent_le_cli.handmanagerocket_websit    awa             _text()
iveecebsocket.rit weawa=     message          
    changes)iption, subscr/pongngmessages (pi client for# Wait                     try:

        ile True:   whe
     on alivtinnecep co    # Ke   
        )
 d, websocketient(user_id_clager.adket_man websocit awa      nager
  to maient Add cl #ry:
       
    t
    eturn   r")
     edliztiat ininager noocket maebSson="Wreacode=1011, ket.close(wait websoc
        anager:_maket not websoc  if
  
    ept()cket.accait webso"
    aw"" updates analytics-timent for realket endpoiWebSoc
    """r):d: str_it, use WebSockecket:bsoint(weocket_endpo websync defr_id}")
as{useytics/ws/analcket("/@app.webso

tr(e))
=sailet500, dtatus_code=ption(sTTPExce   raise H}")
     {e: m chartcustog r creatinrror(f"Erroger.e
        log as e:ptioncexcept Ex e
             }
   rt
   "data": cha            ,
ss": True    "succe
           return {      
 fig)
      dget_condget(witom_wie.create_cus_enginizationvisualart = await 
        ch)
         {}g',onfit.get('cequesart_rconfig = chwidget_            
    
")alizednitingine not iion eualizattail="Vise=503, deod_ctusion(staptTPExcese HT       rai    
 engine:n_zatiosualinot vi  if 
      "
    try:t"" char customeate a""Cr
    ", Any]):[str: Dictequestart_r_chart(chom create_custdef
async custom")ics/charts/analyt("/

@app.post)
l=str(e)e=500, detaitatus_codeption(se HTTPExc  rais")
      oard: {e}shbng dar creatir(f"Errologger.erro      as e:
  eption   except Exc        
          }
ard
bo dashdata":   "         True,
uccess":   "s        eturn {
        r  
            )
  es
  _preferencernfig, usout_co       layyout(
     d_laboareate_dash_widgets.crrdait dashboa = awhboardas   d    
       
   {})s',r_preferenceget('useequest.shboard_rerences = daser_pref      u', {})
  ayout_configst.get('loard_requeashbonfig = d_c     layout 
   )
       d"tializenits not iard widgeDashbotail="3, de_code=50ption(statusxceise HTTPE      rats:
      board_widgesht da    if no    
ry:
    tout"""layrd hboaete dascomple a reat   """Cr, Any]):
 t: Dict[stesboard_requhboard(dash_dascreateasync def 
create")s/dashboard/lytict("/anapos

@app.=str(e))
aile=500, detus_codption(statExce raise HTTP
       e}") widget: {reating c"Errorerror(fr.    logge    as e:
Exception  except           
  }
 
      geta": wid "dat            True,
uccess":   "s         urn {
 ret      
     onfig)
    , ct_type, datat(widgete_widge.crea_widgetsrdashboa dwaitet = aidg 
        w    
   ig', {})get('confequest.= widget_r   config      {})
 a',.get('dat_requestdget  data = wi)
      get('type't.quesreet_idg = wget_type        wid      

  tialized")ets not inird widg="Dashboa, detail03s_code=5on(statu HTTPExceptise         rai:
   _widgetsrddashboaf not      i   try:
   
 "" widget"boardreate a dash"C"]):
    "r, Anyt: Dict[stidget_requese_widget(wef creatasync dcreate")
idgets/tics/wt("/analypp.pos)


@al=str(e) detaie=500,_codtatustion(sse HTTPExceprai    )
    tmap: {e}"ea h portfoliotting(f"Error gegger.error
        loption as e:ce except Ex     
           }

    heatmapta":       "da
     ": True,  "success
           {   return    
         )
_datap(portfoliomaheat_portfolio_atecreengine.sualization_= await vi   heatmap 
     peatmate h# Crea  
          {}
    olio_data = ortf   p     cs engine
nalytit from awould geion, implementatin real data - rtfolio  po # Mock 
       ")
       lizedinitia not inelization enguatail="Vise=503, de(status_codionExcepte HTTP  rais
          tion_engine:ot visualiza    if ntry:
    
    atmap"""formance heolio pert portf    """Ge str):
r_id:_heatmap(usertfolioponc def get_
asy)"r_id}/{uselio-heatmapfoharts/port/analytics/ct(".ge)


@appl=str(e)=500, detaitatus_codetion(sxcep HTTPE  raise
      {e}")shboard:  risk daettingError g".error(fer       loggn as e:
 ioptexcept Exce       
        }
   shboard
  ata": da    "d,
        cess": True      "suc{
          return             
ta)
d(risk_dahboarsk_dascreate_rie.gintion_ent visualizawai aboard =     dash
   dashboard# Create 
           d)
     _is(usersk_metricme_rial_tigine.get_reytics_enalt anaidata = awisk_   r    ta
 sk daGet ri  #   
      )
      "initializedces not l="Servi03, detaius_code=5eption(statHTTPExc    raise       ne:
  tion_engiizavisual not e orlytics_enginnot ana      if :
  
    try""rd"ics dashboarisk metrt "Ge:
    ""str)user_id: rd(sk_dashboaget_ridef ync ")
as_id}isk/{userarts/ranalytics/chget("/
@app.=str(e))

500, detaile=od(status_contiepaise HTTPExc
        r {e}")rt:ance chaormting perfror getErer.error(f"  logg  :
     as exceptioncept Eex
            }
     chart
    ":ata    "d
        : True,ccess"    "su       return {
 
                nce_data)
rt(performaformance_chaerate_pe.creenginzation_ali await visut = char    t
   eate char       # Cr
        
        )ime
 ime=end_t     end_t,
       tart_timert_time=s       sta_id,
     r_id=user   use       metrics(
  mance_perforet_e.gytics_engin analta = awaitrformance_da      pe 
  )
       s=30ta(dayme - timedel_ti_time = end    start
    w()tcnome.uti datee =d_tim     en  ata
 mance dGet perfor     #      
    ")
  izedinitial not vicesdetail="Ser_code=503, ption(statusExcese HTTP       rai     
ion_engine: visualizat or not_enginelytics  if not ana charnalytics ce a: strer_ide_chart(usormancormance/{usearts/perfe))=str(, detail=500atus_coden(st {ert:hang P&L cr gettiro.error(f"Erogger      l  eption as et Exc    excep


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info"
    )   
     return {
            "success": True,
            "data": attribution_data,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": period_days
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/charts/pnl/{user_id}")
async def get_pnl_chart(
    user_id: str,
    timeframe: str = "1D"
):
    """Get P&L attribution chart"""
    try:
        if not analytics_engine or not visualization_engine:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Get P&L data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        pnl_data = await analytics_engine.get_pnl_attribution(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create chart
        chart = await visualization_engine.create_pnl_chart(pnl_data, timeframe)
        
        return {
            "success": True,
            "data": chart
        }
        
    except Exception as e:
        logger.error(f"Error getting P&L chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/charts/performance/{user_id}")
async def get_performance_chart(user_id: str):
    """Get performance analytics chart"""
    try:
        if not analytics_engine or not visualization_engine:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Get performance data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        
        performance_data = await analytics_engine.get_performance_metrics(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create chart
        chart = await visualization_engine.create_performance_chart(performance_data)
        
        return {
            "success": True,
            "data": chart
        }
        
    except Exception as e:
        logger.error(f"Error getting performance chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/charts/risk/{user_id}")
async def get_risk_dashboard(user_id: str):
    """Get risk metrics dashboard"""
    try:
        if not analytics_engine or not visualization_engine:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Get risk data
        risk_data = await analytics_engine.get_real_time_risk_metrics(user_id)
        
        # Create dashboard
        dashboard = await visualization_engine.create_risk_dashboard(risk_data)
        
        return {
            "success": True,
            "data": dashboard
        }
        
    except Exception as e:
        logger.error(f"Error getting risk dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/charts/portfolio-heatmap/{user_id}")
async def get_portfolio_heatmap(user_id: str):
    """Get portfolio performance heatmap"""
    try:
        if not visualization_engine:
            raise HTTPException(status_code=503, detail="Visualization engine not initialized")
        
        # Mock portfolio data - in real implementation, would get from analytics engine
        portfolio_data = {}
        
        # Create heatmap
        heatmap = await visualization_engine.create_portfolio_heatmap(portfolio_data)
        
        return {
            "success": True,
            "data": heatmap
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/widgets/create")
async def create_widget(widget_request: Dict[str, Any]):
    """Create a dashboard widget"""
    try:
        if not dashboard_widgets:
            raise HTTPException(status_code=503, detail="Dashboard widgets not initialized")
        
        widget_type = widget_request.get('type')
        data = widget_request.get('data', {})
        config = widget_request.get('config', {})
        
        widget = await dashboard_widgets.create_widget(widget_type, data, config)
        
        return {
            "success": True,
            "data": widget
        }
        
    except Exception as e:
        logger.error(f"Error creating widget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/dashboard/create")
async def create_dashboard(dashboard_request: Dict[str, Any]):
    """Create a complete dashboard layout"""
    try:
        if not dashboard_widgets:
            raise HTTPException(status_code=503, detail="Dashboard widgets not initialized")
        
        layout_config = dashboard_request.get('layout_config', {})
        user_preferences = dashboard_request.get('user_preferences', {})
        
        dashboard = await dashboard_widgets.create_dashboard_layout(
            layout_config, user_preferences
        )
        
        return {
            "success": True,
            "data": dashboard
        }
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/charts/custom")
async def create_custom_chart(chart_request: Dict[str, Any]):
    """Create a custom chart"""
    try:
        if not visualization_engine:
            raise HTTPException(status_code=503, detail="Visualization engine not initialized")
        
        widget_config = chart_request.get('config', {})
        
        chart = await visualization_engine.create_custom_widget(widget_config)
        
        return {
            "success": True,
            "data": chart
        }
        
    except Exception as e:
        logger.error(f"Error creating custom chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/reports/generate")
async def generate_report(report_request: Dict[str, Any]):
    """Generate a trading report"""
    try:
        if not report_generator or not analytics_engine:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        report_type = report_request.get('report_type', 'daily')
        user_id = report_request.get('user_id')
        config = report_request.get('config', {})
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Get data for report
        end_time = datetime.utcnow()
        if report_type == 'daily':
            start_time = end_time - timedelta(days=1)
        elif report_type == 'weekly':
            start_time = end_time - timedelta(days=7)
        elif report_type == 'monthly':
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)
        
        # Gather report data
        pnl_data = await analytics_engine.get_pnl_attribution(user_id, start_time, end_time)
        performance_data = await analytics_engine.get_performance_metrics(user_id, start_time, end_time)
        risk_data = await analytics_engine.get_real_time_risk_metrics(user_id)
        
        report_data = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_pnl': pnl_data.get('total_pnl', 0),
            'realized_pnl': pnl_data.get('total_realized_pnl', 0),
            'unrealized_pnl': pnl_data.get('total_unrealized_pnl', 0),
            'trades_count': len(pnl_data.get('trade_pnl', {})),
            'win_rate': performance_data.get('trade_metrics', {}).get('win_rate', 0),
            'best_trade': performance_data.get('trade_metrics', {}).get('best_trade', 0),
            'worst_trade': performance_data.get('trade_metrics', {}).get('worst_trade', 0),
            'sharpe_ratio': performance_data.get('risk_metrics', {}).get('sharpe_ratio', 0),
            'max_drawdown': performance_data.get('drawdown_metrics', {}).get('max_drawdown', 0),
            'var_95': risk_data.get('var_metrics', {}).get('var_95', {}).get('recommended', 0),
            'risk_score': risk_data.get('risk_score', 0),
            'positions': [],  # Would get from data aggregator
            'strategies': []   # Would get from data aggregator
        }
        
        # Generate report
        report_info = await report_generator.generate_report(
            report_type, user_id, report_data, config
        )
        
        return {
            "success": True,
            "data": report_info
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/reports/schedule")
async def schedule_report(schedule_request: Dict[str, Any]):
    """Schedule automated report generation"""
    try:
        if not report_generator:
            raise HTTPException(status_code=503, detail="Report generator not initialized")
        
        schedule_info = await report_generator.schedule_report(schedule_request)
        
        return {
            "success": True,
            "data": schedule_info
        }
        
    except Exception as e:
        logger.error(f"Error scheduling report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/reports/send")
async def send_report(send_request: Dict[str, Any]):
    """Send report to recipients"""
    try:
        if not report_generator:
            raise HTTPException(status_code=503, detail="Report generator not initialized")
        
        report_info = send_request.get('report_info', {})
        recipients = send_request.get('recipients', [])
        delivery_method = send_request.get('delivery_method', 'email')
        
        result = await report_generator.send_report(report_info, recipients, delivery_method)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error sending report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/notifications/send")
async def send_notification(notification_request: Dict[str, Any]):
    """Send notification via multiple channels"""
    try:
        if not notification_service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")
        
        notification_type = notification_request.get('type')
        recipients = notification_request.get('recipients', [])
        data = notification_request.get('data', {})
        channels = notification_request.get('channels', ['email'])
        
        result = await notification_service.send_notification(
            notification_type, recipients, data, channels
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/notifications/bulk")
async def send_bulk_notifications(bulk_request: Dict[str, Any]):
    """Send multiple notifications"""
    try:
        if not notification_service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")
        
        notifications = bulk_request.get('notifications', [])
        
        results = await notification_service.send_bulk_notifications(notifications)
        
        return {
            "success": True,
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Error sending bulk notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/notifications/schedule")
async def schedule_notification(schedule_request: Dict[str, Any]):
    """Schedule a notification for future delivery"""
    try:
        if not notification_service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")
        
        schedule_info = await notification_service.schedule_notification(schedule_request)
        
        return {
            "success": True,
            "data": schedule_info
        }
        
    except Exception as e:
        logger.error(f"Error scheduling notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/analytics/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time analytics updates"""
    await websocket.accept()
    
    if not websocket_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return
    
    try:
        # Add client to manager
        await websocket_manager.add_client(user_id, websocket)
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong, subscription changes)
                message = await websocket.receive_text()
                await websocket_manager.handle_client_message(user_id, message)
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        # Remove client from manager
        if websocket_manager:
            await websocket_manager.remove_client(user_id)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info"
    )