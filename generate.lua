local utf8 = require 'lua-utf8'
local Map = require 'pl.Map'
local Set = require 'pl.Set'

config = require 'config'

langs = config.langs

local print_mode = false

for _, ftype in ipairs({'train','valid','test'}) do
  filename = paths.concat(config.data_path, config.filename[ftype])

  if print_mode then
    sample_size = 100
  else
    if ftype == 'train' then
      sample_size = 100000
    elseif ftype == 'valid' then
      sample_size = 1000
    else
      sample_size = 1000
    end
  end

  local terms = {}
  for _, lang in ipairs(langs) do
    terms[lang] = {}

    if lang == 'en' then
      terms['en']['conjunction'] = {'however,','but','therefore,','so','hence'}
      terms['en']['subject'] = {'i','he','she','they','you','taehoon','janghoon'}
      terms['en']['verb'] = {'like','love','hate','go after','catch up with','catch','go to','ask'}
      terms['en']['object'] = {'me','him','her','them','you','taehoon','janghoon'}
    elseif lang == 'ko' then
      terms['ko']['conjunction'] = {'하지만','하지만','그러므로','그래서','그러므로'}
      terms['ko']['subject'] = {'나','그','그녀','그들','너','태훈','장훈'}
      terms['ko']['verb'] = {'좋아한다','사랑한다','싫어한다','따라갔다','따라갔다','잡았다','갔다','물어봤다'}
      terms['ko']['object'] = {'나','그','그녀','그들','너','태훈','장훈'}
    elseif lang == 'ja' then
      terms['ko']['conjunction'] = {'しかし,','だって','だから,','それで','だから'}
      terms['ko']['subject'] = {'私','彼','彼女','태훈','장훈'}
      terms['ko']['verb'] = {'善よがる','愛している','싫어한다','따라갔다','따라갔다','잡았다','갔다','물어봤다'}
      terms['ko']['object'] = {'私','彼','彼女','태훈','장훈'}
    end
  end

  function get_v_o(s, v, o, l, is_past, is_and)
    if l == 'en' then
      if o == 'me' and string.sub(v,1,2) == 'go' then 
        v = string.gsub(v, 'go','come')
      end
      if Set{'he','she','taehoon','janghoon'}[s] and not is_past then
        if string.sub(v,1,2) == 'go' then
          v = string.gsub(v, 'go','goes')
        elseif string.sub(v,1,4) == 'come' then
          v = string.gsub(v, 'come','comes')
        elseif string.sub(v,1,5) == 'catch' then
          v = string.gsub(v, 'catch','catches')
        else
          v = v .. 's'
        end
      end

      if is_past then
        if string.sub(v,1,2) == 'go' then
          v = string.gsub(v, 'go','went')
        elseif string.sub(v,1,4) == 'come' then
          v = string.gsub(v, 'come','came')
        elseif string.sub(v,1,5) == 'catch' then
          v = string.gsub(v, 'catch','catched')
        elseif string.sub(v, -1) == 'e' then
          v = v .. 'd'
        else
          v = v .. 'ed'
        end
      end
    elseif l == 'ko' then
      if o == '나' and utf8.find(v, '갔다') then
        v = utf8.gsub(v, '갔다','왔다')
      end

      if Set{'태훈','장훈','그들'}[s] then s = s .. '은' else s = s .. '는' end
      if Set{'갔다','왔다','물어봤다'}[v] then o = o .. '에게'
      elseif Set{'태훈','장훈','그들'}[o] then o = o .. '을' else o = o .. '를' end

      if is_past then
        if utf8.sub(v, -2) == '한다' then
          v = utf8.sub(v, 1, utf8.len(v)-2) .. '했었다'
        else
          v = utf8.sub(v, 1, utf8.len(v)-1) .. '었다'
        end
      end
      if is_and then
        if utf8.sub(v, -2) == '한다' then
          v = utf8.sub(v, 1, utf8.len(v)-2) .. '하고'
        else
          v = utf8.sub(v, 1, utf8.len(v)-1) .. '고'
        end
      end
    end

    return s, v, o
  end

  function mode(mode_size)
    return math.random(mode_size) % mode_size + 1
  end

  function rand(table)
    return table[math.random(#table)]
  end

  function make(idx,l,c,s,v,o,add_sentence,add_conjunct,is_past)
    if l == 'en' then
      s, v, o = get_v_o(s, v, o, l, is_past, add_sentence)
      str = s..' '..v..' '..o
    elseif l == 'ko' then
      s, v, o = get_v_o(s, v, o, l, is_past, add_sentence)
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

  if not print_mode then
    file = io.open (filename,'w+')
    io.output(file)
  end

  for i=1, sample_size do
    add_conjunct = math.random() > 0.5
    add_sentence = math.random() > 0.5

    ci = math.random(#terms[langs[1]]['conjunction'])
    si = math.random(#terms[langs[1]]['subject'])
    vi = math.random(#terms[langs[1]]['verb'])
    oi = math.random(#terms[langs[1]]['object'])
    is_past = math.random() > 0.5

    while si == oi do oi = math.random(#terms[langs[1]]['object']) end

    if add_sentence then
      ci2 = math.random(#terms[langs[1]]['conjunction'])
      si2 = math.random(#terms[langs[1]]['subject'])
      vi2 = math.random(#terms[langs[1]]['verb'])
      oi2 = math.random(#terms[langs[1]]['object'])
      is_past2 = math.random() > 0.5

      while si2 == oi2 do oi2 = math.random(#terms[langs[1]]['object']) end
    end

    for _, l in ipairs(langs) do
      term = terms[l]
      c, s, v, o =  term.conjunction[ci], term.subject[si], term.verb[vi], term.object[oi]
      str = make(1,l,c,s,v,o,add_sentence,add_conjunct,is_past)

      if add_sentence then
        s, v, o =  term.subject[si2], term.verb[vi2], term.object[oi2]
        str = str .. make(2,l,c,s,v,o,false,false,is_past2)
      end

      if print_mode then
        print(str)
      else
        io.write(str..'\n')
      end
    end
  end

  if not print_mode then
    io.close()
  end
end
