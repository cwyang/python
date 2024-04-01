def process(input_string):
    output = list(input_string)  # Initialize output list to collect processed text
    
    # Pattern to match ANSI sequences, OSC commands ending with BEL, and the insert blank character sequence
    pattern = re.compile(
        r'\x1b\[([0-9;]*)[a-zA-Z]'  # ANSI sequences
        r'|\x1b\]0;.*?\x07'  # OSC command for setting window title, ending with BEL
        r'|\x1b\[([0-9]+)@'  # Insert n blank spaces
    )
    
    last_index = 0  # Track the last index processed
    end_pos = 0
    cur_pos = 0
    for match in pattern.finditer(input_string):
        segment = input_string[last_index:match.start()]
        for char in segment:
            if char == '\r':
                cur_pos = 0
            elif char == '\b':
                cur_pos = max(0, cur_pos - 1)
            else:
                output[cur_pos] = char
                cur_pos += 1
                end_pos = max(end_pos, cur_pos)
        
        sequence = match.group(0)
        if sequence.startswith('\x1b]0;') and sequence.endswith('\x07'):
            # OSC command to set window title, ignore/remove it
            pass
        elif sequence.endswith('@'):
            # Insert n blank spaces
            n = int(match.group(2))  # Number of spaces to insert
            output[cur_pos:cur_pos] = ' ' * n
            end_pos += n
        elif sequence[-1] == 'P':  # DCH - Delete character(s)
            n = int(match.group(1)) if match.group(1) else 1
            output[cur_pos:cur_pos+n] = [ ]
            end_pos = max (0, end_pos - n)
        elif sequence[-1] == 'm':  # SGR - Set/reset attribute
            pass
        elif sequence[-1] == 'C':  # CUF - Cursor forward
            n = int(match.group(1)) if match.group(1) else 1
            print(n)
            cur_pos += n
            end_pos = max(end_pos, cur_pos)
        
        last_index = match.end()
    
    # Add the remaining part of the string, processing backspaces
    segment = input_string[last_index:]
    for char in segment:
        if char == '\r':
            cur_pos = 0
        elif char == '\b':
            cur_pos = max(0, cur_pos - 1)
        else:
            output[cur_pos] = char
            cur_pos += 1
            end_pos = max(end_pos, cur_pos)
    
    return ''.join(output[:end_pos])

sample='''
Script started on 2024-04-01 19:17:40+09:00 [TERM="xterm-256color" TTY="/dev/pts/2" COLUMNS="252" LINES="67"]
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ echo hi\r
hi\r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ e\r[4@(reverse-i-search)`':[C[Cc': echo hi[1@h[C[C[C[C\r[6P]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$\r]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ [C\r
hi\r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ exit\r

Script done on 2024-04-01 19:17:46+09:00 [COMMAND_EXIT_CODE="0"]
[01;32mcwyang@Mistral[00m:[01;34m~[00m$ e\r[4@(reverse-i-search)`':[C[Cc': echo hi[1@h[C[C[C[C\r[6P]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$\r]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ [C\r
[01;32mcwyang@Mistral[00m:[01;34m~[00m$ e\r[4@(reverse-i-search)`':[C[Cc': echo hi[1@h[C[C[C[C\r[6P]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$\r]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ [C\r
'''

sample2='''
Script started on 2024-04-01 21:46:50+09:00 [TERM="xterm-256color" TTY="/dev/pts/3" COLUMNS="280" LINES="70"]
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ echo "hi"\r
hi\r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ ec\r
ec2metadata  echo         \r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ ec2metadata \r
^CTraceback (most recent call last):\r
  File "/usr/bin/ec2metadata", line 249, in <module>\r
    main()\r
  File "/usr/bin/ec2metadata", line 245, in main\r
    display(metaopts, burl, prefix)\r
  File "/usr/bin/ec2metadata", line 190, in display\r
    m = EC2Metadata(burl)\r
  File "/usr/bin/ec2metadata", line 117, in __init__\r
    if not self._test_connectivity(addr, port):\r
  File "/usr/bin/ec2metadata", line 126, in _test_connectivity\r
    s.connect((addr, port))\r
KeyboardInterrupt\r
\r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ ec2metadata [3Pho "hi"2metadata [K[K[K[K[K[K[K[K[K[K[K[K\r[4@(reverse-i-search)`':[Ce': echo "hi"[1@c[C[C[C2': ec2metadata \r[7P]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$\r]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ \r
^CTraceback (most recent call last):\r
  File "/usr/bin/ec2metadata", line 249, in <module>\r
    main()\r
  File "/usr/bin/ec2metadata", line 245, in main\r
    display(metaopts, burl, prefix)\r
  File "/usr/bin/ec2metadata", line 190, in display\r
    m = EC2Metadata(burl)\r
  File "/usr/bin/ec2metadata", line 117, in __init__\r
    if not self._test_connectivity(addr, port):\r
  File "/usr/bin/ec2metadata", line 126, in _test_connectivity\r
    s.connect((addr, port))\r
KeyboardInterrupt\r
\r
]0;cwyang@Mistral: ~[01;32mcwyang@Mistral[00m:[01;34m~[00m$ exit\r

Script done on 2024-04-01 21:47:08+09:00 [COMMAND_EXIT_CODE="130"]
'''
for line in sample.split('\n'):
    print(process(line))
