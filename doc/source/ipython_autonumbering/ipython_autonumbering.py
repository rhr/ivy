def ipython_autonumber(app, docname, source):
    import re
    global count
    count = 0
    def repl_numbers(matchobj):
        if matchobj.group()[0] == ' ':
            global count
            count += 1
        return (matchobj.group()[0] + "[" + str(count) + "]")

    source[0] = re.sub(".?\[\*\]", repl_numbers, source[0])

         

def setup(app):
	app.connect('source-read', ipython_autonumber)
	app.add_config_value('ipython_autonumber_include', True, True)
