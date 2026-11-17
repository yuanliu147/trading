"""从本轮实测表和审查结论生成自包含报告/中文Excel，不造数。"""
from pathlib import Path
import html
import json
import pandas as pd
import numpy as np
from openpyxl.styles import Font, PatternFill, Alignment

ROOT=Path(__file__).resolve().parents[1]


def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,list):return [clean_json(v) for v in x]
    if isinstance(x,(np.integer,np.floating)):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def main():
    summary=json.loads((ROOT/'证据/本轮结论与验收.json').read_text())
    maps=pd.read_csv(ROOT/'结果/层级尺度地图.csv')
    bigsmall=pd.read_csv(ROOT/'结果/指数大小周期网格.csv')
    files=list(sorted((ROOT/'结果').glob('*.csv')))
    # 大型逐股票诊断保留CSV入口，不灌进Excel导致巨大内存/体积。
    entries=[]
    out=ROOT/'板块指数大小周期_数据表.xlsx'
    with pd.ExcelWriter(out,engine='openpyxl') as writer:
        pd.DataFrame(summary['结论表']).to_excel(writer,sheet_name='结论与限制',index=False)
        used={'结论与限制'}
        for p in files:
            if p.name=='尺度图逐标的.csv':
                entries.append({'文件':p.name,'行数':'逐标的完整诊断','工作表':'仅CSV'});continue
            df=pd.read_csv(p)
            if len(df)>100000:
                entries.append({'文件':p.name,'行数':len(df),'工作表':'仅CSV'});continue
            name=p.stem[:31];base=name
            i=1
            while name in used:
                i+=1;name=base[:27]+'_'+str(i)
            used.add(name)
            df.to_excel(writer,sheet_name=name,index=False)
            entries.append({'文件':p.name,'行数':len(df),'工作表':name})
        for sh in writer.book:
            sh.freeze_panes='A2';sh.auto_filter.ref=sh.dimensions
            for cell in sh[1]:
                cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='173B45')
                cell.alignment=Alignment(wrap_text=True,vertical='center')
                sh.column_dimensions[cell.column_letter].width=min(35,max(13,len(str(cell.value))*1.4))
            sh.row_dimensions[1].height=38
            for col in range(1,sh.max_column+1):
                label=str(sh.cell(1,col).value)
                ratio=any(k in label for k in ['年化','回撤','仓位','胜率','收益','超额','比例','覆盖率','下界','上界','费用','摩擦'])
                already_percent=any(k in label for k in ['(%)','（%）','百分点','(bp)'])
                if ratio:
                    for row in range(2,sh.max_row+1):
                        cell=sh.cell(row,col)
                        if isinstance(cell.value,(int,float)):
                            cell.number_format='0.00;[Red]-0.00' if already_percent else '0.00%;[Red]-0.00%'
    index=pd.DataFrame(entries)
    index.to_csv(ROOT/'证据/交付表清单.csv',index=False,encoding='utf-8-sig')
    payload=clean_json(maps[['层级','窗口','聚合日数','标的总数','可评估标的数','方差比中位数','方差比P25','方差比P75']].to_dict('records'))
    bullets=''.join('<li>'+html.escape(x)+'</li>' for x in summary['核心说明'])
    limits=''.join('<li>'+html.escape(x)+'</li>' for x in summary['限制'])
    findings=pd.DataFrame(summary['结论表']).to_html(index=False,border=0,escape=True)
    known=bigsmall[bigsmall['单边费用'].eq(.0005)&bigsmall['大周期'].eq(120)&bigsmall['小周期'].eq(20)]
    columns=['标的','组合动作','净年化','最大回撤','平均仓位','超大周期同暴露年化对数差','超大周期同暴露95下界']
    example=known[columns].copy()
    for c in columns[2:]:example[c]=example[c].map(lambda x:f'{x:+.2%}' if pd.notna(x) else '样本不足')
    example=example.to_html(index=False,border=0)
    index_example=pd.DataFrame(summary['指数120日案例'])
    for col in index_example:
        if '(%)' in col:index_example[col]=index_example[col].map(lambda v:f'{v:+.2f}%' if pd.notna(v) else 'NA')
    index_example=index_example.to_html(index=False,border=0)
    links=''.join(f'<li><a href="结果/{p.name}">{html.escape(p.stem)}</a></li>' for p in files)
    sections=''.join(f'<h2>{html.escape(k)}</h2><p>{html.escape(v)}</p>' for k,v in summary['研究解释'].items())
    text="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>板块趋势、指数趋势与大小周期</title><style>
    :root{--ink:#173b45;--muted:#536c74;--border:#d8e0e2;--paper:#fff;--bg:#f2f5f4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1080px;margin:auto;padding:42px 22px 80px}h1{font-size:30px;margin:0 0 8px}h2{font-size:20px;margin:32px 0 10px}p{margin:8px 0 15px}.muted{color:var(--muted);font-size:13px}.lead{border-left:4px solid #be791e;background:#fff7e8;padding:14px 18px;margin:22px 0}.panel{background:var(--paper);padding:16px 18px;margin:16px 0;border:1px solid var(--border);border-radius:8px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:right}th{background:#eaf0ef}td:first-child,th:first-child{text-align:left}select{padding:8px 10px;margin:8px 8px 8px 0;max-width:100%}label{font-size:13px}button{padding:8px 12px}#map td{min-width:70px;text-align:center}#map td:first-child{text-align:left}a{color:#236478}li{margin:5px 0}details{margin:15px 0}footer{margin-top:30px;color:var(--muted);font-size:13px}tr:hover td{outline:1px solid #b9ced2}.plain{white-space:pre-wrap}.small{font-size:12px}@media(max-width:600px){main{padding:24px 12px 60px}h1{font-size:24px}th,td{padding:6px}}@media print{body{background:white}main{padding:10px}select{display:none}.panel{break-inside:avoid}a{color:inherit}}
    </style></head><body><main><h1>板块趋势、指数趋势与大小周期</h1><p class="muted">2010—2020 描述性研究 · 行业主窗2015—2020 · 2021以后未评估 · 指数价格代理不等于ETF实盘收益</p><div class="lead">__HEADLINE__</div><ul>__BULLETS__</ul><div class="panel scroll">__FINDINGS__</div><h2>层级 × 时间尺度</h2><p>数值是方差比：大于1表示该样本路径有延续倾向，小于1表示回复倾向；不是收益率，更不保证策略能赚钱。不同层级和时间窗不能混读。</p><label for="window">选择窗口</label><select id="window"></select><div class="panel scroll"><table id="map"></table></div><p id="mapnote" class="muted" aria-live="polite"></p><p class="muted">官方行业缓存存在缺口，NA表示连续样本不足5倍聚合日数；完整行业等权合成作为补充，不把缺失日压缩为连续交易日。详细实际样本起止见尺度图逐标的CSV。</p>__SECTIONS__<h2>指数120日均线：效果因指数而异</h2><p>预先登记的同一规则、10bp主成本。指数价格代理，不含股息或实际ETF追踪误差；表中百分数已经按百分数单位显示。不能把沪深300的正例当成所有指数都适用。</p><div class="panel scroll">__INDEXEXAMPLE__</div><h2>一个预先登记的大小周期例子</h2><p>120日大均线 × 20日小均线，单边5bp代理成本。列出全部7资产两种动作，不把它称为最佳参数。“大多小弱”指大均线上、小均线下立即持有，不是底部确认。</p><div class="panel scroll">__EXAMPLE__</div><h2>方法更正与可执行边界</h2><ul>__LIMITS__</ul><h2>完整数据与复现</h2><p><a href="板块指数大小周期_数据表.xlsx">中文表头Excel</a> · <a href="证据/本轮结论与验收.json">本轮结论与验收</a> · <a href="证据/研究设计冻结.json">冻结研究设计</a> · <a href="证据/补充登记与前轮更正.json">追加登记与旧结论更正</a></p><details><summary>全部CSV结果</summary><ul>__LINKS__</ul></details><p class="muted">复现：依次运行代码/01_构建层级数据.py、02_时序与行业轮动.py、03_层级尺度图.py、04_跨层条件检验.py、05_指数大小周期.py、08_跨层敏感性.py、09_汇总结论.py、06_报告与表格.py、07_交付检查.py。证据目录保留统计、成本与日期测试。</p><footer>本报告只说明这些明确规则在历史数据中的表现；不是交易建议，不含当前买卖点，不宣称时间样本外验证通过。总体发现是层级/尺度依赖，而不是某一种策略在所有市场都有效或无效。</footer></main><script>
    const DATA=__DATA__;const sel=document.getElementById('window'), map=document.getElementById('map'), note=document.getElementById('mapnote');const periods=[...new Set(DATA.map(x=>x['窗口']))];periods.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;sel.appendChild(o)});sel.value='行业主窗2015-2020';
    function draw(){const group=DATA.filter(x=>x['窗口']===sel.value),levels=[...new Set(group.map(x=>x['层级']))];const qs=[2,5,10,20,60,120,250];map.replaceChildren();let tr=document.createElement('tr');['层级',...qs.map(q=>q+'交易日')].forEach(s=>{let t=document.createElement('th');t.textContent=s;tr.appendChild(t)});map.appendChild(tr);levels.forEach(level=>{tr=document.createElement('tr');let label=document.createElement('td');label.textContent=level;tr.appendChild(label);qs.forEach(q=>{const row=group.find(x=>x['层级']===level&&x['聚合日数']===q),cell=document.createElement('td');const value=row?.['方差比中位数'];if(value==null){cell.textContent='NA';cell.title='样本不足，不可评估'}else{cell.textContent=value.toFixed(3);cell.style.background=value>1?'rgba(51,131,118,'+Math.min(.4,(value-1)*.5+.04)+')':'rgba(201,122,53,'+Math.min(.4,(1-value)*.5+.04)+')';cell.title='可评估 '+row['可评估标的数']+' / '+row['标的总数']+'；P25 '+(row['方差比P25']??0).toFixed(3)+'，P75 '+(row['方差比P75']??0).toFixed(3)}tr.appendChild(cell)});map.appendChild(tr)});note.textContent='当前：'+sel.value+'。色深只表示离1的距离，不表示统计显著性。行业缓存结果采用最长完整段，不能当全窗无缺失结果。'}sel.addEventListener('change',draw);draw();
    </script></body></html>"""
    replacements={'__HEADLINE__':html.escape(summary['总判断']),'__BULLETS__':bullets,'__FINDINGS__':findings,
      '__SECTIONS__':sections,'__INDEXEXAMPLE__':index_example,'__EXAMPLE__':example,'__LIMITS__':limits,'__LINKS__':links,
      '__DATA__':json.dumps(payload,ensure_ascii=False,allow_nan=False)}
    for k,v in replacements.items():text=text.replace(k,v)
    p=ROOT/'板块指数大小周期_研究报告.html';p.write_text(text)
    print('报告',p,'Excel',out,'表数',len(used),'HTML字节',p.stat().st_size)

if __name__=='__main__':main()
