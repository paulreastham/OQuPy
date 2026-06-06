#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input::Initialization:: *)
elsol[inispl_?NumericQ,betaomegac_?NumericQ]:=Module[{eqns},eqns=1/(2 \[Omega]c r[t]^2) E^(r[t]/\[Omega]c) Sech[1/2 \[Beta] r[t]]^4 (\[Omega]c Sinh[\[Beta] r[t]] Derivative[1][r][t]^2+r[t] ((-2 \[Beta] \[Omega]c+\[Beta] \[Omega]c Cosh[\[Beta] r[t]]-Sinh[\[Beta] r[t]]) Derivative[1][r][t]^2-2 \[Omega]c Sinh[\[Beta] r[t]] Derivative[2][r][t]))/. {\[Beta]->1,\[Omega]c->betaomegac}; 
(*Solve the equations in dimensionless form*);
res[inisl_?NumericQ]:=r/. NDSolve[{eqns==0,r[0]==inispl,r'[0]==inisl},r,{t,0,1}][[1]];solvedforisl=a/. FindRoot[res[a][1]==0,{a,0}];res[solvedforisl]];
optimaltdlensol[temp_,inisplit_,tf_,omegac_,times_]:=Module[{t=N[temp],ini=N[inisplit],duration=N[tf],oc=N[omegac],ts,dimensionlessresult},ts=N@Flatten@{times};dimensionlessresult=elsol[ini/t,oc/t];Developer`ToPackedArray[t*(N[dimensionlessresult[#/duration]])&/@ts]];


(* ::Input::Initialization:: *)
thermstate[split_,temp_]:=Module[{z,pup,pdn},z=2*Cosh[split/(2*temp)];
pup=Exp[-split/(2*temp)]/z;
pdn=Exp[+split/(2*temp)]/z;
{{pup,0},{0,pdn}}];


(* ::Input::Initialization:: *)
ham[w_]:={{w/2,0},{0,-w/2}};
id=IdentityMatrix[2];
sp={{0,1},{0,0}};
sm=Transpose[sp];


(* ::Input::Initialization:: *)
lop[o_]:=KroneckerProduct[o,id];rop[o_]:=KroneckerProduct[id,Transpose[o]];
lrop[o1_,o2_]:=KroneckerProduct[o1,Transpose[o2]];
dissipator[op_]:=Module[{opd,opdop},
opd=Transpose[Conjugate[op]];
opdop=opd . op;
lrop[op,opd]-(lop[opdop]+rop[opdop])/2];


(* ::Input::Initialization:: *)
(* Determine the optimal Lindblad solution via thermodynamic length, and the dynamics of sx and heat in that solution *)
optsol[temp_,inisplit_,tf_,omegac_,dt_,alpha_,times_]:=Module[{},splits=optimaltdlensol[temp,inisplit,tf,omegac,times];
j[w_]:=2*alpha*w*Exp[-w/omegac];nb[w_]:=If[temp>0, 1/(Exp[w/temp]-1),0];dnrate[w_]:=(Pi/2)*j[w]*(1+nb[w]);uprate[w_]:=(Pi/2)*j[w]*nb[w];
diss[w_]:=uprate[w]*dissipator[sp]+dnrate[w]*dissipator[sm];
lvprop[step_]:=Module[{},hamstep=ham[splits[[step]]];hamprop=-I*(lop[hamstep]-rop[hamstep]);hamprop=MatrixExp[dt*hamprop];dissprop=MatrixExp[dt*diss[splits[[step]]]];dissprop . hamprop];
states={};
rhoini=thermstate[inisplit,temp];
state=Flatten[rhoini];props=Table[lvprop[step],{step,1,Length[times]}];For[step=1,step<Length[props],step++;AppendTo[states,state];state=props[[step]] . state];AppendTo[states,state];rhooft=ArrayReshape[states,{Length[states],2,2}];sx=(Tr[{{1,0},{0,-1}} . #])& /@ rhooft;hrateop[w_]:=w*{{dnrate[w],0},{0,-uprate[w]}};hrateops=hrateop /@ splits;hcurrent=MapThread[Tr[#1 . #2]&,{hrateops,rhooft}];hperstep=hcurrent*dt;qoft=Re[Accumulate[hperstep]]; N[qoft]];

temp=0.1
inisplit=1.0
tf=400
omegac=1.0
dt=0.2
alpha=0.05
times=Table[t,{t,0,tf,dt}]
optsol[temp,inisplit,tf,omegac,dt,alpha,times]






