require 'utf8sub'

Map = require 'pl.Map'
Set = require 'pl.Set'

sample_size = 10000

terms = {}
langs = {'en','ko'}

for _, lang in ipairs(langs) do
  terms[lang] = {}
end

terms['en']['conjunction'] = {'however','but','therefore','so','hence'}
terms['ko']['conjunction'] = {'하지만','하지만','그러므로','그래서','그러므로'}
terms['en']['subject'] = {'i','he','she','taehoon','janghoon'}
terms['ko']['subject'] = {'나','그','그녀','태훈','장훈'}
terms['en']['verb'] = {'like','liked','love','loved','hate','hated','go after','went after','catch up with','catch','catched','go to','went to','ask','asked'}
terms['ko']['verb'] = {'좋아한다','좋아했다','사랑한다','사랑했다','싫어한다','싫어했다','따라갔다','따라갔었다','따라갔다','잡았다','잡았었다','갔다','갔었다','물어봤다','물어봤었다'}
terms['ko']['verb2'] = {'좋아하고','좋아했고','사랑하고','사랑했고','싫어하고','싫어했고','따라갔고','따라갔었고','따라갔고','잡았고','잡았었고','갔고','갔었고','물어봤고','물어봤었고'}
terms['en']['object'] = {'me','him','her','taehoon','janghoon'}
terms['ko']['object'] = {'나','그','그녀','태훈','장훈'}

function mode(mode_size)
  return math.random(mode_size) % mode_size + 1
end

function rand(table)
  return table[math.random(#table)]
end

function make(idx,l,c,s,v,o,term,add_sentence,add_conjunct)
  if l == 'en' then
    if Set{'he','she','taehoon','janghoon'}[s] then
      if string.sub(v,-1) ~= 'd' and string.sub(v,-5) ~= 'after' and string.sub(v,-4) ~= 'with' and string.sub(v,-2) ~= 'to' then
        if Set{'catch'}[v] then
          v = v .. 'es'
        else
          v = v .. 's'
        end
      elseif string.sub(v,1,2) == 'go' then
        v = string.gsub(v, 'go','goes')
      elseif string.sub(v,1,5) == 'catch' and string.sub(v,1,7) ~= 'catched' then
        v = string.gsub(v, 'catch','catches')
      end
    end
    str = s..' '..v..' '..o
  elseif l == 'ko' then
    if Set{'태훈','장훈'}[s] then s = s .. '은' else s = s .. '는' end
    if Set{'갔다','갔었다','갔고','갔었고','물어봤다','물어봤었다','물어봤고','물어봤었고'}[v] then o = o .. '에게'
    elseif Set{'태훈','장훈'}[o] then o = o .. '을' else o = o .. '를' end

    if add_sentence and idx == 1 then
      v = term.verb2[vi]
    end

    str = s..' '..o..' '..v
  end

  if add_conjunct and idx == 1 then
    str = c..' '..str
  end

  if idx == 2 then
    if l == 'en' then str = ' and '..str
    elseif l== 'ko' then str = ' '..str
    end
  end

  return str
end

file = io.open (string.format('%s_%s.txt',langs[1],langs[2]),'w+')
io.output(file)

for i=1, sample_size do
  add_conjunct = math.random() > 0.5
  add_sentence = math.random() > 0.5

  ci = math.random(#terms[langs[1]]['conjunction'])
  si = math.random(#terms[langs[1]]['subject'])
  vi = math.random(#terms[langs[1]]['verb'])
  oi = math.random(#terms[langs[1]]['object'])
  if add_sentence then
    ci2 = math.random(#terms[langs[1]]['conjunction'])
    si2 = math.random(#terms[langs[1]]['subject'])
    vi2 = math.random(#terms[langs[1]]['verb'])
    oi2 = math.random(#terms[langs[1]]['object'])
  end

  for _, l in ipairs(langs) do
    term = terms[l]
    c, s, v, o =  term.conjunction[ci], term.subject[si], term.verb[vi], term.object[oi]
    str = make(1,l,c,s,v,o,term,add_sentence, add_conjunct)

    if add_sentence then
      s, v, o =  term.subject[si2], term.verb[vi2], term.object[oi2]
      str = str .. make(2,l,c,s,v,o,term)
    end

    io.write(str..'\n')
  end
end
io.close()
