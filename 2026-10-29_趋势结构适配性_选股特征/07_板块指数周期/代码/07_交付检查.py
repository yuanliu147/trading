"""本轮交付核查：核对实际产物、边界、内部链接与工作表，不替代方法审查。"""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    def test(name,ok,evidence):rows.append({'检查':name,'通过':bool(ok),'证据':evidence})
    design=ROOT/'证据/研究设计冻结.json'
    audit=json.loads((ROOT/'证据/数据核验.json').read_text())
    expected=next(v for k,v in audit['源文件哈希'].items() if k.endswith('研究设计冻结.json'))
    test('原设计冻结哈希未变',hashlib.sha256(design.read_bytes()).hexdigest()==expected,str(design))
    d=pd.read_csv(ROOT/'结果/尺度图逐标的.csv',usecols=['层级','窗口','聚合日数','连续可用日数','方差比','实测段止'])
    test('尺度图有效样本n>=5q',not (d.方差比.notna()&d.连续可用日数.lt(d.聚合日数*5)).any(),str(len(d)))
    test('尺度图止于2020',d.实测段止.dropna().le('2020-12-31').all(),'实测段止最大='+str(d.实测段止.dropna().max()))
    test('行业归属不读取未来生效记录',audit['行业映射使用未来记录行数']==0,str(audit['行业映射使用未来记录行数']))
    combo=pd.read_csv(ROOT/'结果/指数大小周期网格.csv')
    test('大小周期全网格252格',len(combo)==252,str(len(combo)))
    test('大小周期评价止于2020',combo.评价末日.le('2020-12-31').all(),str(combo.评价末日.max()))
    report=ROOT/'板块指数大小周期_研究报告.html'
    txt=report.read_text();dom=BeautifulSoup(txt,'html.parser')
    test('报告生成无模板占位',not re.search(r'__[A-Z]+__',txt),'模板替换')
    test('报告可读且注明描述性',bool(dom.find('h1')) and '描述性' in dom.get_text(),dom.find('h1').get_text())
    test('报告注明代理非ETF','不等于ETF' in dom.get_text(),'价格指数代理声明')
    missing=[]
    for a in dom.select('a[href]'):
        link=a['href']
        if link.startswith(('http:','https:','#')):continue
        if not (report.parent/link).exists():missing.append(link)
    test('所有本地结果链接存在',not missing,str(missing))
    wb=load_workbook(ROOT/'板块指数大小周期_数据表.xlsx',read_only=True,data_only=True)
    test('Excel包含核心尺度图和大小周期',{'层级尺度地图','指数大小周期网格'}.issubset(wb.sheetnames),str(wb.sheetnames))
    test('Excel至少六个工作表',len(wb.sheetnames)>=6,'工作表数='+str(len(wb.sheetnames)))
    test('Excel结论不是空白',wb['结论与限制'].max_row>3,'行数='+str(wb['结论与限制'].max_row))
    wb.close()
    summary=json.loads((ROOT/'证据/本轮结论与验收.json').read_text())
    test('三个问题逐项有解释',len(summary['研究解释'])>=3,str(list(summary['研究解释'])))
    test('重要限制有记录',len(summary['限制'])>=5,str(len(summary['限制'])))
    out={'检查项':rows,'通过数':sum(x['通过'] for x in rows),'检查数':len(rows),
         '说明':'这些检查不验证策略泛化；独立方法审查及实际统计结果须一起读。不把检查通过等同所有假说成立。'}
    (ROOT/'证据/交付检查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    for row in rows:print(('PASS' if row['通过'] else 'FAIL'),row['检查'],row['证据'])
    assert all(row['通过'] for row in rows),'仍有未通过项目'

if __name__=='__main__':main()
