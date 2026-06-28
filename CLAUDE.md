# CLAUDE.md

# trying out the code
Normally the user already has the chained-surprisal-distributions/ app running at http://localhost:5006/ which serves up chained-surprisal-distributions/main.py.  Note the user started this with "cd chained-surprisal-distributions/ && find . | grep \.py$ | entr -r bokeh serve . >> chained-surprisal-distributions.stdout.txt 2>> chained-surprisal-distributions.stderr.txt" so any logging should immediately be reflected in those two files, both of which are in entropy-sim/chained-surprisal-distributions/.  BTW "entr" just watches those .py files and restarts bokeh as needed.

After editing main.py, entr reliably restarts the server immediately. Do not check server logs to confirm the restart, do not grep for new PIDs, do not use browser_wait_for — just navigate with Playwright and test. If the page doesn't look right, the cause is never a loading delay — investigate why the content isn't there.

In the stderr after you make a change you'll see 'ProtocolError("Token is expired. Configure the app with a larger value for --session-token-expiration if necessary")'  Those are from stale browser tabs trying to reconnect to it. Ignore.

# timeouts
Whatever you are doing, don't set a timeout on it longer than 2 seconds and if you EVER find yourself waiting on something, TELL THE USER LOUDLY about it. Something is wrong and the user wants to know about that.

# subagents
I don't have any experience using subagents, so I'm relying on you to spot opportunities to use them. Let me know if you see a good opportunity!
