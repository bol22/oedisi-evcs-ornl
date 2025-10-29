from fastapi import FastAPI, BackgroundTasks, HTTPException
from evcs_federate import run_simulator
from oedisi.types.common import BrokerConfig
from fastapi.responses import JSONResponse
import traceback
import socket
import json
import os
from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames

app = FastAPI()

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()
    return JSONResponse(response, 200)

@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulator, broker_config)
        resp = ServerReply(detail="Task successfully added.").dict()
        return JSONResponse(resp, 200)
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())

@app.post("/configure")
async def configure(component_struct: ComponentStruct):
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {link.target_port: f"{link.source}/{link.source_port}" for link in component_struct.links}
    json.dump(links, open(DefaultFileNames.INPUT_MAPPING.value, "w"))
    json.dump(params, open(DefaultFileNames.STATIC_INPUTS.value, "w"))
    return JSONResponse(ServerReply(detail="Configuration updated.").dict(), 200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))