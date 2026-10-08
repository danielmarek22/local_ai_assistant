import json
from types import SimpleNamespace
from unittest.mock import Mock
from app.autonomy.store import AutonomyStore
from app.core.response_generator import ResponseGenerator
from app.core.tool_executor import ToolExecutor
from app.core.context_builder import ContextBuilder
from app.integrations import CapabilityId, RegisteredTool, ToolSpec, ToolResult, IntegrationRegistry
from app.memory.chat_history import ChatHistoryStore
from app.memory.summary_store import SummaryStore
from app.storage.database import Database

out={}
called=[]
class Demo:
 name='demo'
 def registered_tools(self):
  return [RegisteredTool(ToolSpec(CapabilityId('demo','blocked'),'Harmless probe',{'type':'object','properties':{},'additionalProperties':False}), lambda args,ctx: (called.append('called') or ToolResult.success('probe')))]
registry=IntegrationRegistry([Demo()])
responses=iter([{'tool_calls':[{'function':{'name':'demo__blocked','arguments':{}}}]},{'content':'Done'}])
class Model:
 def chat_buffered(self,**kwargs):
  out.setdefault('schemas_seen',[]).append(kwargs['tools'])
  return next(responses)
g=ResponseGenerator(Model(),ToolExecutor(registry),lambda *a,**kw:None)
list(g.stream_late_routed_response('probe', [{'role':'user','content':'test'}], 'test', allowed_capabilities=set(),persist_tool_traces=False))
out['empty_allowlist_handler_executed']=bool(called)

s=AutonomyStore(':memory:')
s.begin_operation('op','demo__blocked','probe',None,None,None)
s.finish_operation('op','success','first terminal result')
s.finish_operation('op','error','late contradictory result')
out['terminal_status_after_success_then_error']=s.get_operation('op').status
s.close()

db=Database(':memory:')
h=ChatHistoryStore(db,SimpleNamespace(episodic_collection=Mock()))
ss=SummaryStore(db)
with db.transaction() as c:
 for i in range(100): c.execute("INSERT INTO chat_history(session_id,role,content) VALUES('other','user',?)",(f'other-{i}',))
 c.execute("INSERT INTO chat_history(session_id,role,content) VALUES('target','user','summarized')")
 checkpoint=c.execute('SELECT MAX(id) FROM chat_history').fetchone()[0]
 for i in range(4): c.execute("INSERT INTO chat_history(session_id,role,content) VALUES('target','user',?)",(f'new-{i}',))
ss.set('target','Previous summary',checkpoint)
b=ContextBuilder('System',h,history_limit=10,summary_store=ss)
messages=b.build('target','current')
out['checkpoint']=checkpoint
out['unsummarized_expected']=['new-0','new-1','new-2','new-3']
out['unsummarized_in_context']=[m['content'] for m in messages if m['role']=='user']
db.close()
print(json.dumps(out,indent=2))
