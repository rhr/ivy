"""
Unittests for SSE methods
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
import numpy as np
from ivy.chars.sse import sse
from ivy.chars import discrete
import nlopt

class sse_methods(unittest.TestCase):
    def setUp(self):
        self.threetiptree = ivy.tree.read(u'((A:1,B:1):1,C:2)root;')
        self.hundredtiptree = ivy.tree.read("(((((sp57:4.044339169,(sp58:3.890662993,sp59:3.890662993)nd118:0.1536761762)nd111:1.417401845,((((sp75:3.172705141,(sp125:0.5527391639,sp126:0.5527391639)nd128:2.619965977)nd126:0.1491783771,sp71:3.321883518)nd121:0.1716653213,(sp111:1.329916304,sp112:1.329916304)nd122:2.163632536)nd119:0.5173899544,(sp62:3.482885335,sp63:3.482885335)nd120:0.5280534591)nd112:1.45080222)nd28:16.04986342,(((sp52:4.711305107,sp53:4.711305107)nd98:2.079117401,sp44:6.790422508)nd58:5.683724767,sp21:12.47414727)nd14:9.03745716)nd10:13.84492974,(((((sp81:2.783684725,(sp96:1.931056014,sp97:1.931056014)nd132:0.8526287114)nd55:9.9533422,sp18:12.73702692)nd38:1.512996403,((((((sp64:3.470751646,sp65:3.470751646)nd113:1.344122386,sp50:4.814874033)nd89:3.976689789,(sp69:3.328192332,sp70:3.328192332)nd90:5.46337149)nd88:0.02754570427,sp31:8.819109526)nd80:0.653847391,(sp79:2.817908157,(sp129:0.370641198,sp130:0.370641198)nd130:2.447266959)nd116:6.65504876)nd53:3.303754679,((sp38:9.117749002,(sp133:0.2778588591,sp134:0.2778588591)nd83:8.839890143)nd77:3.012840068,((sp137:0.07400931118,sp138:0.07400931118)nd74:10.10434534,(sp36:6.895546513,sp105:6.895546513)nd75:3.282808135)nd60:1.952234422)nd54:0.6461225269)nd39:1.473311731)nd26:4.269769285,(sp17:14.65578684,(sp54:4.704808503,((sp84:2.568626404,sp85:2.568626404)nd123:0.8723411983,(((sp101:1.714268281,sp102:1.714268281)nd137:0.02759929337,sp100:1.741867575)nd135:0.643384511,sp88:2.385252086)nd124:1.055715516)nd115:1.263840901)nd56:9.950978337)nd27:3.864005773)nd25:3.392164376,(sp14:15.11915802,((sp42:5.948314005,(sp98:1.843935789,sp99:1.843935789)nd109:4.104378217)nd66:4.996946603,(sp66:7.141750109,(sp48:5.383149397,sp49:5.383149397)nd93:1.758600712)nd67:3.803510499)nd35:4.173897408)nd31:6.792798972)nd8:13.44457719)nd5:10.70015921,(((((sp45:5.958477059,(sp107:1.503900566,sp108:1.503900566)nd108:4.454576494)nd64:5.427227553,(sp86:2.522861611,sp87:2.522861611)nd65:8.862843002)nd63:5.960631133,sp10:17.34633575)nd24:2.32921451,(((sp103:1.63925749,sp104:1.63925749)nd61:12.27804272,(((sp68:3.331939432,((sp94:1.994874638,sp95:1.994874638)nd131:0.7907986681,sp80:2.785673306)nd127:0.5462661258)nd110:9.657761967,(sp20:12.61435609,((sp115:1.093273853,sp116:1.093273853)nd102:5.49578194,sp37:6.589055793)nd57:6.025300299)nd52:0.3753453071)nd46:0.78118111,(sp60:3.853622154,sp61:3.853622154)nd48:9.917260354)nd43:0.1464177026)nd22:5.428430873,(((sp113:1.329796843,sp114:1.329796843)nd94:8.61369875,sp27:9.943495594)nd70:4.226538594,(((sp92:2.005629918,sp93:2.005629918)nd68:8.934427814,(((sp123:0.7745931871,sp124:0.7745931871)nd84:8.202637625,(sp89:2.308927308,(sp131:0.3602063283,sp132:0.3602063283)nd136:1.94872098)nd85:6.668303504)nd71:1.471438122,sp26:10.44866893)nd69:0.4913887978)nd49:2.389357191,((sp76:3.070054683,sp77:3.070054683)nd105:3.747331878,(((sp127:0.5455466061,sp128:0.5455466061)nd106:5.417574288,sp41:5.963120894)nd100:0.7261380398,(sp117:1.071580803,sp118:1.071580803)nd101:5.61767813)nd97:0.1281276275)nd50:6.512028362)nd41:0.8406192645)nd23:5.175696897)nd19:0.3298191716)nd16:0.5359322381,((((sp90:2.193767491,sp91:2.193767491)nd78:7.555963912,(sp121:0.85277711,sp122:0.85277711)nd79:8.896954293)nd32:6.289306951,((((sp109:1.37117621,sp110:1.37117621)nd133:1.33986203,(sp135:0.1976309559,sp136:0.1976309559)nd134:2.513407283)nd129:0.162954131,sp78:2.87399237)nd103:3.38367749,(sp72:3.248274386,sp73:3.248274386)nd104:3.009395474)nd33:9.781368494)nd20:3.487554657,(sp56:10.38265213,(sp119:0.9775309416,sp120:0.9775309416)nd114:9.405121191)nd21:9.143940879)nd17:0.6848894819)nd6:25.84521089)nd1;")
        self.twentytiptree = ivy.tree.read("((((sp10:6.01550249,sp11:6.01550249)nd15:4.554189871,sp6:10.56969236)nd4:5.673112954,((sp17:1.761858488,sp18:1.761858488)nd18:14.28585769,(sp1:14.22233298,((sp19:1.192115887,(sp20:0.9593031603,sp21:0.9593031603)nd24:0.232812727)nd17:12.20230714,(sp12:5.966802788,((sp15:2.751773764,sp16:2.751773764)nd20:1.509183886,((sp24:0.5210034045,(sp25:0.4522778413,sp26:0.4522778413)nd25:0.06872556325)nd22:2.307862297,(sp22:0.8313908298,sp23:0.8313908298)nd23:1.997474871)nd21:1.432091949)nd19:1.705845138)nd10:7.427620235)nd8:0.8279099527)nd7:1.825383202)nd5:0.1950891369)nd2:0.2486838349,((sp13:5.001447697,sp14:5.001447697)nd16:5.570676616,sp5:10.57212431)nd13:5.919364837)nd1;")

        self.threetipdata = {"A":1, "B":1,"C":0}
        self.hundredtipdata = {'sp10':0,'sp14':0,'sp17':0,'sp18':0,'sp20':1,'sp21':1,'sp26':0,'sp27':0,'sp31':0,'sp36':0,'sp37':0,'sp38':1,'sp41':1,'sp42':0,'sp44':0,'sp45':1,'sp48':0,'sp49':0,'sp50':0,'sp52':1,'sp53':0,'sp54':1,'sp56':0,'sp57':0,'sp58':1,'sp59':1,'sp60':1,'sp61':0,'sp62':1,'sp63':1,'sp64':1,'sp65':0,'sp66':0,'sp68':1,'sp69':0,'sp70':1,'sp71':0,'sp72':0,'sp73':0,'sp75':0,'sp76':0,'sp77':0,'sp78':0,'sp79':1,'sp80':1,'sp81':0,'sp84':0,'sp85':0,'sp86':0,'sp87':0,'sp88':1,'sp89':0,'sp90':1,'sp91':1,'sp92':0,'sp93':0,'sp94':1,'sp95':1,'sp96':1,'sp97':1,'sp98':1,'sp99':0,'sp100':1,'sp101':1,'sp102':1,'sp103':1,'sp104':1,'sp105':1,'sp107':0,'sp108':1,'sp109':1,'sp110':1,'sp111':0,'sp112':0,'sp113':0,'sp114':0,'sp115':0,'sp116':0,'sp117':1,'sp118':1,'sp119':1,'sp120':0,'sp121':0,'sp122':0,'sp123':0,'sp124':0,'sp125':1,'sp126':1,'sp127':0,'sp128':0,'sp129':0,'sp130':0,'sp131':0,'sp132':0,'sp133':1,'sp134':1,'sp135':1,'sp136':1,'sp137':1,'sp138':1}
        self.twentytipdata = {'sp1':0,'sp5':1,'sp6':0,'sp10':1,'sp11':0,'sp12':0,'sp13':0,'sp14':0,'sp15':0,'sp16':0,'sp17':0,'sp18':0,'sp19':1,'sp20':1,'sp21':1,'sp22':0,'sp23':0,'sp24':1,'sp25':1,'sp26':1}
    def test_bisse_threetip_unequalparams(self):
        root = self.threetiptree
        data = self.threetipdata

        params = np.array([0.1, 0.2, 0.01, 0.02, 0.05, 0.07])

        calc_lik = sse.bisse_odeiv(root,data,params,condition_on_surv=True)
        true_lik = -5.007626
        self.assertTrue(np.isclose(calc_lik,true_lik,atol=1e-7))
    def test_classe_threetip_twostate(self):
        root = self.threetiptree
        data = self.threetipdata
        lambdaidx = np.array([[[1,2],
                               [0,3]],
                              [[4,5],
                               [0,6]]])
        muidx = np.array([7,8])
        qidx = np.array([[0,9],
                         [10,0]])
        pidx = {"lambda":lambdaidx,"mu":muidx,"q":qidx}
        f = sse.make_classe(root,data,2,pidx,False,pi="Equal")
        params = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])

        calc_lik_ncos = f(params)
        true_lik_ncos = -5.962547

        f = sse.make_classe(root,data,2,pidx,True,pi="Equal")
        params = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])
        calc_lik_cos = f(params)
        true_lik_cos = -4.912378
        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1e-3))
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1e-3))

    def test_classe_threetip_threestate(self):
        root = self.threetiptree
        data = self.threetipdata

        f = sse.make_classe(root,data,nstate=3,condition_on_surv=False,pi="Equal")
        params = np.array([0.3,0.1,0.7,0.01,0.8,0.05,0.01,0.1,0.9,0.2,0.004,0.3,0.3,0.11,0.14,0.15,0.17,0.19,0.01,0.01,0.1,0.2,0.4,0.1,0.01,0.7,0.6])
        calc_lik_ncos = f(params)
        true_lik_ncos = -9.6298

        f = sse.make_classe(root,data,nstate=3,condition_on_surv=True,pi="Equal")
        calc_lik_cos = f(params)
        true_lik_cos = -10.00689
        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1e-3))
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1e-3))
# #
    def test_classe_20tip2state(self):
        root = self.twentytiptree
        data = self.twentytipdata
        params = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])

        f = sse.make_classe(root,data,2,None,False,pi="Equal")
        calc_lik_ncos = f(params)
        true_lik_ncos = -86.35204
        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1e-2))

        f = sse.make_classe(root,data,2,None,True,pi="Equal")
        calc_lik_cos = f(params)
        true_lik_cos = -85.28472
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1e-2))

    def test_classe_100tip_2state(self):
        root = self.hundredtiptree
        data = self.hundredtipdata

        params = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])

        f = sse.make_classe(root,data,2,None,False,pi="Equal")
        calc_lik_ncos = f(params)
        true_lik_ncos = -452.8292

        f = sse.make_classe(root,data,2,None,True,pi="Equal")
        calc_lik_cos = f(params)
        print(calc_lik_cos)
        true_lik_cos = -451.7423

        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1e-1))
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1e-1))
    def test_classe_100tip_geosseequiv(self):
        root = self.hundredtiptree
        data = self.hundredtipdata

        lambdaidx = np.array([[[0,1,2],
                     [0,0,3],
                     [0,0,0]],
                    [[0,0,0],
                     [0,1,0],
                     [0,0,0]],
                    [[0,0,0],
                     [0,0,0],
                     [0,0,2]]])
        muidx = np.array([0,4,5])
        qidx = np.array([[0,5,4],
                         [6,0,0],
                         [7,0,0]])
        pidx = {"lambda":lambdaidx,"mu":muidx,"q":qidx}

        params = np.array([0.157769502155153, 0.157769502155153, 0.157769502155153,
                           0.0788847510775767, 0.0788847510775767, 0.0788847510775767, 0.0788847510775767])

        f = sse.make_classe(root,data,nstate=3,pidx=pidx)
        calc_lik = f(params)
        true_lik = -440.9617
        self.assertTrue(np.isclose(calc_lik,true_lik,atol=1e-1))

    def test_classe_100tip_2state_MLE(self):
        root = self.hundredtiptree
        data = self.hundredtipdata

        startingvals = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])
        out = sse.fit_classe(root,data)
        optim = sse.param_dict_to_list(out["fit"])

        r_optim = [1.068944e-01, 2.988239e-08, 7.893486e-02, 9.706393e-06, 3.305960e-07, 6.974496e-02, 4.384240e-02, 3.603397e-08, 5.317747e-06, 1.539999e-01]
        self.assertTrue(np.isclose(optim,r_optim,atol=0.1).all())

class sse_calculations(unittest.TestCase):
    def setUp(self):
        self.params2state = ["lambda000","lambda001","lambda011","lambda100","lambda101","lambda111","mu0","mu1","q01","q10"]
        self.params3state = ["lambda000","lambda001","lambda002","lambda011","lambda012","lambda022",
                             "lambda100","lambda101","lambda102","lambda111","lambda112","lambda122",
                             "lambda200","lambda201","lambda202","lambda211","lambda212","lambda222",
                             "mu0","mu1","mu2","q01","q02","q10","q12","q20","q21"]
    def test_unpackparams2state_lambda(self):
        lambda000 = discrete.sse_get_lambda(self.params2state,0,0,0,2)
        lambda001 = discrete.sse_get_lambda(self.params2state,0,0,1,2)
        lambda010 = discrete.sse_get_lambda(self.params2state,0,1,0,2)
        lambda011 = discrete.sse_get_lambda(self.params2state,0,1,1,2)
        lambda100 = discrete.sse_get_lambda(self.params2state,1,0,0,2)
        lambda101 = discrete.sse_get_lambda(self.params2state,1,0,1,2)
        lambda110 = discrete.sse_get_lambda(self.params2state,1,1,0,2)
        lambda111 = discrete.sse_get_lambda(self.params2state,1,1,1,2)

        self.assertEquals(lambda000,"lambda000")
        self.assertEquals(lambda001,"lambda001")
        self.assertEquals(lambda010,0)
        self.assertEquals(lambda100,"lambda100")
        self.assertEquals(lambda101,"lambda101")
        self.assertEquals(lambda110,0)
        self.assertEquals(lambda111,"lambda111")

    def test_unpackparams2state_mu(self):
        mu0 = discrete.sse_get_mu(self.params2state,0,2)
        mu1 = discrete.sse_get_mu(self.params2state,1,2)

        self.assertEquals(mu0,"mu0")
        self.assertEquals(mu1,"mu1")
    def test_unpackparams2state_q(self):
        q00 = discrete.sse_get_qij(self.params2state,0,0,2)
        q01 = discrete.sse_get_qij(self.params2state,0,1,2)
        q10 = discrete.sse_get_qij(self.params2state,1,0,2)
        q11 = discrete.sse_get_qij(self.params2state,1,1,2)

        self.assertEquals(q00,0)
        self.assertEquals(q01,"q01")
        self.assertEquals(q10,"q10")
        self.assertEquals(q11,0)
    def test_unpackparams3state_lambda(self):
        lambda000 = discrete.sse_get_lambda(self.params3state,0,0,0,3)
        lambda001 = discrete.sse_get_lambda(self.params3state,0,0,1,3)
        lambda002 = discrete.sse_get_lambda(self.params3state,0,0,2,3)
        lambda010 = discrete.sse_get_lambda(self.params3state,0,1,0,3)
        lambda011 = discrete.sse_get_lambda(self.params3state,0,1,1,3)
        lambda012 = discrete.sse_get_lambda(self.params3state,0,1,2,3)
        lambda020 = discrete.sse_get_lambda(self.params3state,0,2,0,3)
        lambda021 = discrete.sse_get_lambda(self.params3state,0,2,1,3)
        lambda022 = discrete.sse_get_lambda(self.params3state,0,2,2,3)
        lambda100 = discrete.sse_get_lambda(self.params3state,1,0,0,3)
        lambda101 = discrete.sse_get_lambda(self.params3state,1,0,1,3)
        lambda102 = discrete.sse_get_lambda(self.params3state,1,0,2,3)
        lambda110 = discrete.sse_get_lambda(self.params3state,1,1,0,3)
        lambda111 = discrete.sse_get_lambda(self.params3state,1,1,1,3)
        lambda112 = discrete.sse_get_lambda(self.params3state,1,1,2,3)
        lambda120 = discrete.sse_get_lambda(self.params3state,1,2,0,3)
        lambda121 = discrete.sse_get_lambda(self.params3state,1,2,1,3)
        lambda122 = discrete.sse_get_lambda(self.params3state,1,2,2,3)
        lambda200 = discrete.sse_get_lambda(self.params3state,2,0,0,3)
        lambda201 = discrete.sse_get_lambda(self.params3state,2,0,1,3)
        lambda202 = discrete.sse_get_lambda(self.params3state,2,0,2,3)
        lambda210 = discrete.sse_get_lambda(self.params3state,2,1,0,3)
        lambda211 = discrete.sse_get_lambda(self.params3state,2,1,1,3)
        lambda212 = discrete.sse_get_lambda(self.params3state,2,1,2,3)
        lambda220 = discrete.sse_get_lambda(self.params3state,2,2,0,3)
        lambda221 = discrete.sse_get_lambda(self.params3state,2,2,1,3)
        lambda222 = discrete.sse_get_lambda(self.params3state,2,2,2,3)

        self.assertEquals(lambda000,"lambda000")
        self.assertEquals(lambda001,"lambda001")
        self.assertEquals(lambda002,"lambda002")
        self.assertEquals(lambda010,0)
        self.assertEquals(lambda011,"lambda011")
        self.assertEquals(lambda012,"lambda012")
        self.assertEquals(lambda020,0)
        self.assertEquals(lambda021,0)
        self.assertEquals(lambda022,"lambda022")

        self.assertEquals(lambda100,"lambda100")
        self.assertEquals(lambda101,"lambda101")
        self.assertEquals(lambda102,"lambda102")
        self.assertEquals(lambda110,0)
        self.assertEquals(lambda111,"lambda111")
        self.assertEquals(lambda112,"lambda112")
        self.assertEquals(lambda120,0)
        self.assertEquals(lambda121,0)
        self.assertEquals(lambda122,"lambda122")

        self.assertEquals(lambda200,"lambda200")
        self.assertEquals(lambda201,"lambda201")
        self.assertEquals(lambda202,"lambda202")
        self.assertEquals(lambda210,0)
        self.assertEquals(lambda211,"lambda211")
        self.assertEquals(lambda212,"lambda212")
        self.assertEquals(lambda220,0)
        self.assertEquals(lambda221,0)
        self.assertEquals(lambda222,"lambda222")
    def test_unpackparams3state_mu(self):
        mu0 = discrete.sse_get_mu(self.params3state,0,3)
        mu1 = discrete.sse_get_mu(self.params3state,1,3)
        mu2 = discrete.sse_get_mu(self.params3state,2,3)

        self.assertEquals(mu0,"mu0")
        self.assertEquals(mu1,"mu1")
        self.assertEquals(mu2,"mu2")
    def test_unpackparams3state_q(self):
        q00 = discrete.sse_get_qij(self.params3state,0,0,3)
        q01 = discrete.sse_get_qij(self.params3state,0,1,3)
        q02 = discrete.sse_get_qij(self.params3state,0,2,3)

        q10 = discrete.sse_get_qij(self.params3state,1,0,3)
        q11 = discrete.sse_get_qij(self.params3state,1,1,3)
        q12 = discrete.sse_get_qij(self.params3state,1,2,3)

        q20 = discrete.sse_get_qij(self.params3state,2,0,3)
        q21 = discrete.sse_get_qij(self.params3state,2,1,3)
        q22 = discrete.sse_get_qij(self.params3state,2,2,3)


        self.assertEquals(q00,0)
        self.assertEquals(q01,"q01")
        self.assertEquals(q02,"q02")
        self.assertEquals(q10,"q10")
        self.assertEquals(q11,0)
        self.assertEquals(q12,"q12")
        self.assertEquals(q20,"q20")
        self.assertEquals(q21,"q21")
        self.assertEquals(q22,0)
    def test_processparams_2state(self):
        lambda0 = np.array([[0.3,0.1],
                            [0.0,0.01]])
        lambda1 = np.array([[0.01,0.1],
                            [0.0,0.2]])
        lambdas = np.array([lambda0,lambda1])

        mus = np.array([0.01,0.01])
        q = np.array([[0,0.2],
                      [0.1,0]])

        paramdict = {"lambda":lambdas,"mu":mus,"q":q}
        out = sse.param_dict_to_list(paramdict)
        true_params = np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])

        self.assertTrue((out==true_params).all())
    def test_prettyparams_2state(self):
        lambda0 = np.array([[0.3,0.1],
                            [0.0,0.01]])
        lambda1 = np.array([[0.01,0.1],
                            [0.0,0.2]])
        lambdas = np.array([lambda0,lambda1])

        mus = np.array([0.01,0.01])
        q = np.array([[0,0.2],
                      [0.1,0]])

        trueparamdict = {"lambda":lambdas,"mu":mus,"q":q}
        params=np.array([0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])
        out = sse.param_list_to_dict(params,2)
        self.assertTrue(all([np.isclose(out[x],trueparamdict[x]).all() for x in out.keys()]))
    def test_processparams_3state(self):
        lambda0 = np.array([[0.3,0.1,0.7],
                            [0.0,0.01,0.8],
                            [0,0,0.05]])
        lambda1 =np.array([[0.01,0.1,0.9],
                            [0.0,0.2,0.004],
                            [0,0,0.3]])
        lambda2 =np.array([[0.3,0.11,0.14],
                            [0.0,0.15,0.17],
                            [0,0,0.19]])
        lambdas = np.array([lambda0,lambda1,lambda2])

        mus = np.array([0.01,0.01,0.1])
        q = np.array([[0.0,0.2,0.4],
                      [0.1,0.0,0.01],
                      [0.7,0.6,0.00]])

        paramdict = {"lambda":lambdas,"mu":mus,"q":q}
        out = sse.param_dict_to_list(paramdict)
        true_params = np.array([0.3,0.1,0.7,0.01,0.8,0.05,0.01,0.1,0.9,0.2,0.004,0.3,0.3,0.11,0.14,0.15,0.17,0.19,0.01,0.01,0.1,0.2,0.4,0.1,0.01,0.7,0.6])

        self.assertTrue((out==true_params).all())

if __name__=="__main__":
    unittest.main()
