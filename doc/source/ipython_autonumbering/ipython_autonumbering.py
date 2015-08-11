def ipython_autonumber(app, docname, source):
    import re
    global count
    count = 0
    def repl_numbers(matchobj):
        global count
        if matchobj.group()[0] == ' ':
            count += 1
        if matchobj.group() == "====":
            count = 0
            return("====")
        else:
            return (matchobj.group()[0] + "[" + str(count) + "]")
    source[0] = re.sub(".?\[\*\]|====", repl_numbers, source[0])

         

def setup(app):
	app.connect('source-read', ipython_autonumber)
	app.add_config_value('ipython_autonumber_include', True, True)
