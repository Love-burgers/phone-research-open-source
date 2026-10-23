const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const html = fs.readFileSync(path.join(__dirname, '../app/static/index.html'), 'utf8');
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(inline);
function helpers(fetch) {
  const context = vm.createContext({fetch, URL});
  vm.runInContext(inline.slice(0, inline.indexOf('// UI starts here.')), context);
  return context;
}

test('frontend and backend share the prose/details behavior contract',()=>{
  const {testGapCopy}=helpers();
  const cases=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/presentation_copy_cases.json'),'utf8'));
  for(const sample of cases){
    const shown=testGapCopy(sample.text,true);
    assert.equal(shown.text,sample.expected);
    assert.equal(Boolean(shown.notices.length),sample.notice);
  }
});

test('access token stays in a tab session and is only attached to local API URLs',async()=>{
  const calls=[];
  const context=helpers(async(url,options)=>{calls.push({url,options});return {ok:true,status:200,json:async()=>({status:'ok'})};});
  context.sessionStorage={getItem:()=> 'test-access'};
  await context.chatRequest('/api/chat/sessions');
  await context.chatRequest('https://example.com/');
  assert.equal(calls[0].options.headers.Authorization,'Bearer test-access');
  assert.equal(calls[1].options.headers.Authorization,undefined);
  assert.match(html,/id="accessToken" type="password"/);
  assert.doesNotMatch(html,/localStorage\.setItem\(['"]phone-selection-access/);
});

test('personal model key stays in memory and only accompanies chat inference',async()=>{
  const calls=[];
  const context=helpers(async(url,options)=>{calls.push({url,options});return {ok:true,status:200,json:async()=>({status:'ok'})};});
  context.sessionStorage={getItem:()=>null,setItem:()=>assert.fail('key persisted')};
  context.localStorage={setItem:()=>assert.fail('key persisted')};
  context.setModelApiKey('offline-personal-key');
  assert.equal(context.hasModelApiKey(),true);
  for(const url of ['/api/chat','/api/health','/api/chat/sessions','https://example.com/api/chat']) await context.chatRequest(url);
  assert.equal(calls[0].options.headers['X-Model-API-Key'],'offline-personal-key');
  for(const call of calls.slice(1)) assert.equal(call.options.headers['X-Model-API-Key'],undefined);
  context.clearModelApiKey();
  await context.chatRequest('/api/chat');
  assert.equal(calls[4].options.headers['X-Model-API-Key'],undefined);
  assert.equal(helpers().hasModelApiKey(),false);
  assert.match(html,/id="modelApiKey" type="password"/);
  assert.match(html,/!latestHealth.model_configured && !hasModelApiKey\(\)/);
});

test('personal key validation never echoes input and save does not test the provider',()=>{
  const context=helpers();
  assert.throws(()=>context.setModelApiKey('secret invalid credential'),/API Key格式无效/);
  assert.equal(context.hasModelApiKey(),false);
  const submit=inline.slice(inline.indexOf("$('#modelKeyForm').addEventListener"),inline.indexOf("$('#skipModelKey').addEventListener"));
  assert.doesNotMatch(submit,/chatRequest|fetch\(|localStorage|sessionStorage/);
  assert.match(submit,/location.protocol/);
});

test('network button explicitly selects local or public search in the request', () => {
  const {chatPayload}=helpers();
  assert.equal(chatPayload('也想兼顾拍照','a'.repeat(32),false).search_mode,'local');
  assert.equal(chatPayload('换一批',null,true).search_mode,'web');
  assert.equal(chatPayload('换一批',null,true).message,'换一批');
  assert.match(html,/id="webSearch" aria-pressed="false"/);
  assert.match(html,/id="mode" hidden/);
});

test('generic test-gap notices move into details without erasing specifications or measured results',()=>{
  const {recommendationCopy,testGapCopy,pendingMatch}=helpers();
  const shown=recommendationCopy('骁龙8至尊版、6000mAh电池。价格5999元在预算内，但游戏与续航实测待补。',true);
  assert.equal(shown.text,'骁龙8至尊版、6000mAh电池。价格5999元在预算内。');
  assert.match(shown.notices.join(''),/实测待补/);
  assert.equal(testGapCopy('骁龙8，游戏表现待补实测。',true).text,'骁龙8。');
  const measured='原神最高画质连续30分钟，平均59帧，温度40℃。';
  assert.equal(testGapCopy(measured,true).text,measured);
  assert.equal(testGapCopy('暂无对应实测。',false).text,'暂无对应实测。');
  assert.equal(pendingMatch('原神：待查'),true);
  assert.equal(pendingMatch('屏幕：配置不符合'),false);
});

test('recommendation copy moves price gaps into details without losing phone facts', () => {
  const {recommendationCopy} = helpers();
  const text='5000万像素长焦，支持OIS。80W有线闪充加50W无线闪充；版本价未核对到，能否在5000元内落地仍待确认。本轮未查到版本价，是否符合5000元预算需另行核对。';
  const shown=recommendationCopy(text,true);
  assert.equal(shown.text,'5000万像素长焦，支持OIS。80W有线闪充加50W无线闪充。');
  assert.equal(shown.notices.length,2);
  assert.match(shown.notices.join(''),/版本价未核对到/);
});

test('recommendation copy keeps real quotes, budget breaches and source conflicts', () => {
  const {recommendationCopy} = helpers();
  for (const text of ['官网标价16GB+256GB 4999元。','该版本售价5299元，超预算299元，优惠未确认。','两个来源同版本价格冲突，需核对。','国行与港版价格未确认。','版本价未查到，但已核对80W快充。']) {
    assert.equal(recommendationCopy(text,true).text,text);
    assert.equal(recommendationCopy(text,true).notices.length,0);
  }
});

test('direct price questions and warning-only results remain visible', () => {
  const {recommendationCopy} = helpers();
  const text='目前未查到版本售价。';
  assert.equal(recommendationCopy(text,false).text,text);
  assert.equal(recommendationCopy(text,true).text,text);
});

test('source summary counts cited webpages separately from uncited lookup records', () => {
  const {evidenceGroups}=helpers();
  const message={blocks:[{evidence_ids:['spec','price']}],sources:[
    {id:'spec',url:'https://example.com/product'},{id:'price',url:'https://example.com/product'},
    ...Array.from({length:23},(_,i)=>({id:'preview'+i,url:'https://example.com/search/'+i}))]};
  const groups=evidenceGroups(message);
  assert.equal(groups.cited.length,1);assert.equal(groups.cited[0].sources.length,2);
  assert.equal(groups.other.length,23);assert.equal(message.sources.length,25);
});
test('mode distinguishes completed network work, cache reuse and no network', () => {
  const {chatModeText} = helpers();
  const status = metrics => chatModeText({}, 'AI对话', {engine:'langgraph-chat-agent', metrics});
  assert.match(status({network_calls:3, cache_hits:1}), /本轮联网查询3次/);
  assert.match(status({network_calls:0, cache_hits:2}), /复用查询缓存/);
  assert.match(status({network_calls:0, cache_hits:0}), /本轮未联网/);
});
test('historical execution boundary is not shown as a provider outage', () => {
  const {chatModeText} = helpers();
  const trace = {engine:'langgraph-chat-agent',metrics:{network_calls:2},actions:[{action:'report',status:'skipped',reason:'token_budget'}]};
  assert.match(chatModeText({},'资料模式（模型暂不可用）',trace),/资料参考/);
  trace.actions.push({action:'decision',status:'error',error_type:'TimeoutError'});
  assert.match(chatModeText({},'资料模式（模型暂不可用）',trace),/模型暂不可用/);
});
test('chat source URLs reject executable and authenticated URLs', () => {
  const {safeUrl} = helpers();
  assert.equal(safeUrl('javascript:alert(1)'),null);
  assert.equal(safeUrl('https://user:secret@example.com'),null);
  assert.equal(safeUrl('https://example.com/phone'),'https://example.com/phone');
});
test('chat helper keeps conflict messages but hides internal errors', () => {
  const {chatApiError} = helpers();
  assert.match(chatApiError({detail:'这段对话已收到另一条消息'},409),/另一条消息/);
  assert.equal(chatApiError({detail:'private'},500),'服务暂不可用，请稍后重试。');
  assert.match(chatApiError({detail:[{}]},422),/输入/);
});
test('empty delete response and session response are handled separately', async () => {
  const {chatRequest} = helpers(async () => ({ok:true,status:204}));
  assert.equal(await chatRequest('/session',{method:'DELETE'}),null);
  const context = helpers(async () => ({ok:true,status:200,json:async () => null}));
  await assert.rejects(context.chatRequest('/session'),/格式异常/);
});

test('actual reply takes precedence over stale health on send or restore', () => {
  const {chatModeText} = helpers();
  assert.match(chatModeText({model_configured:false},'AI对话'),/^AI对话/);
  assert.match(chatModeText({model_configured:true},'资料模式（模型暂不可用）'),/^资料模式（模型暂不可用）/);
  assert.match(chatModeText({model_configured:true},'资料模式（回答未通过核对）'),/回答未通过核对.*失败记录/);
  assert.match(chatModeText({model_configured:true},'资料模式'),/^资料模式/);
  assert.match(chatModeText({model_configured:true}),/^AI对话/);
  assert.match(chatModeText({model_configured:false}),/^资料模式/);
  assert.match(chatModeText({model_configured:true},'AI对话',{engine:'langgraph-chat-agent'}),/^AI Agent/);
  assert.match(chatModeText({model_configured:true},'资料模式',{engine:'langgraph-chat-agent'}),/^AI Agent.*待核对/);
});

test('sources identify body, snippet and knowledge with date and market', () => {
  const {sourceTypeText} = helpers();
  assert.match(sourceTypeText({source_type:'snippet',checked_on:'2026-10-11',market:'地区未确认'}),/搜索摘要.*正文未核对.*2026-10-11.*地区未确认/);
  assert.match(sourceTypeText({source_type:'page',checked_on:'2026-10-11'}),/已抓取正文/);
  assert.match(sourceTypeText({source_type:'knowledge',checked_on:'2026-10-11'}),/本地知识库/);
  assert.match(sourceTypeText({source_type:'knowledge',evidence_kind:'review',published_on:'2025-09-26',checked_on:'2026-10-11'}),/本地知识库.*实测评测.*2025-09-26发布.*2026-10-11核对/);
  assert.match(sourceTypeText({source_type:'knowledge',evidence_kind:'price',checked_on:'2026-10-11',market:'中国大陆'}),/价格记录.*中国大陆/);
});
