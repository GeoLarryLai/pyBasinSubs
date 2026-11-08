# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Node-based DOM unit test for the 1-D controls (not browser automation)."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(not shutil.which('node'), reason='optional Node JS syntax/DOM unit check')
def test_age_proportional_slider_snaps_and_invalidates():
    source = Path(__file__).resolve().parents[1]/'pybasinsubs/webapp/static/app.js'
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
 constructor(){this.value='';this.hidden=false;this.textContent='';this.classList={toggle(){}};}
 querySelectorAll(){return [];}
 addEventListener(){}
 setAttribute(k,v){this[k]=v;}
 replaceChildren(...items){this.children=items;}
}
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},querySelectorAll(){return [];}};
const context={document,console,window:{},Option:class{constructor(label,value){this.label=label;this.value=value;}},clearInterval(){},setInterval(fn){context.tick=fn;return 1;}};
vm.createContext(context);
let code=fs.readFileSync(process.argv[1],'utf8');
code=code.slice(0,code.indexOf('(async function init()'));
vm.runInContext(code,context);
vm.runInContext(`
const f=(age,i)=>({age_Ma:age,label:age+' Ma',figure:'png'+i,layers:[]});
installColumnFrames('geo',[f(199,0),f(171,1),f(100.5,2),f(0,3)]);
if(+$('c_geo_time').min!==-199||+$('c_geo_time').max!==0)throw Error('age range');
if(+$('c_geo_time').value!==0)throw Error('youngest initial frame');
$('c_geo_time').value=-102;$('c_geo_time').oninput();
if(+$('c_geo_time').value!==-100.5||columnIndices.geo!==2)throw Error('snap to actual age, not frame index');
$('c_geo_time').onkeydown({key:'ArrowLeft',preventDefault(){}});
if(+$('c_geo_time').value!==-171)throw Error('keyboard nodes');
$('c_geo_play').onclick();
if(columnIndices.geo!==0)throw Error('play starts oldest');
`,context);
context.tick();
vm.runInContext(`if(columnIndices.geo!==1)throw Error('play progresses younger');invalidate('c',3);if(columnFrames.geo.length||!$('c_geo_box').hidden||columnTimer!==null)throw Error('stale frames/playback retained');`,context);
vm.runInContext(`
installColumnFrames('thermo',[{...f(199,0),maturity_figure:'m0'},{...f(66,1),maturity_figure:'m1'},{...f(0,2),maturity_figure:'m2'}]);
if($('c_thermo_frame').src!=='data:image/png;base64,png2'||$('c_thermo_maturity').src!=='data:image/png;base64,m2')throw Error('thermal image pair not installed');
$('c_thermo_time').value=-66;$('c_thermo_time').oninput();
if($('c_thermo_frame').src!=='data:image/png;base64,png1'||$('c_thermo_maturity').src!=='data:image/png;base64,m1')throw Error('thermal images not in lockstep');
invalidate('c',4);if(columnFrames.thermo.length||!$('c_thermo_box').hidden)throw Error('stale linked maturity retained');
`,context);
// Equal-age events are available through node buttons even at the same slider position.
vm.runInContext(`installColumnFrames('geo',[f(10,0),f(2.7,1),f(2.7,2),f(0,3)]);showColumnFrame('geo',1);$('c_geo_next').onclick();if(columnIndices.geo!==2||+$('c_geo_time').value!==-2.7)throw Error('erosion event dropped');`,context);
'''
    completed = subprocess.run(['node', '-e', script, str(source)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr



def test_shared_age_controls_above_both_figure_stacks_and_live_maturity_not_hidden():
    from html.parser import HTMLParser
    import re
    root = Path(__file__).resolve().parents[1]/'pybasinsubs/webapp/static'
    class Tags(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack = []; self.parents = {}; self.ids = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.append(attrs['id']); self.parents[attrs['id']] = list(self.stack)
            if tag not in {'input', 'img', 'br', 'meta', 'link', 'hr', 'source'}:
                self.stack.append((tag, attrs.get('id')))
        def handle_endtag(self, tag):
            for i in range(len(self.stack)-1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]; break
    parsed = Tags(); parsed.feed((root/'index.html').read_text())
    assert len(parsed.ids) == len(set(parsed.ids))
    for key in ['geo', 'thermo']:
        assert parsed.ids.index(f'c_{key}_time') < parsed.ids.index(f'c_{key}_play') < parsed.ids.index(f'c_{key}_frame')
        assert not any(parent == 'details' for parent, _ in parsed.parents[f'c_{key}_frame'])
    assert parsed.ids.index('c_thermo_frame') < parsed.ids.index('c_thermo_maturity') < parsed.ids.index('c_thermo_layers') < parsed.ids.index('c_mat')
    assert ('div', 'c_thermo_box') in parsed.parents['c_thermo_maturity']
    assert not any(tag == 'details' for tag, _ in parsed.parents['c_thermo_maturity'])
    assert any(tag == 'details' for tag, _ in parsed.parents['c_mat'])
    css = (root/'app.css').read_text()
    track = re.search(r'\.column-age-track\s*\{([^}]+)\}', css).group(1)
    assert 'position:static' in track and 'absolute' not in track and 'transform' not in track
