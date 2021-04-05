(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{4496:function(e,r,t){"use strict";t.r(r),t.d(r,{default:function(){return I}});var n=t(5893),i=t(336),s=t(4115),o=t(8482),a=t(8017),c=t(7922),l=t(296),p=t(3719),d=t(9444),u=t(7544),m=t(1664),h={1:"Harvard University",2:"Oxford University",3:"Boston Children's Hospital"},x=[{name:"Luke Melas-Kyraizi",institutions:[1,2],url:"https://lukemelas.github.io/"},{name:"Arjun K. Manrai",institutions:[1,3],url:"https://www.childrenshospital.org/research/researchers/m/arjun-manrai"}],f=function(e){var r=e.title;return(0,n.jsxs)(u.Container,{children:[(0,n.jsx)(i.X,{fontSize:"2xl",pt:"3rem",maxW:"42rem",textAlign:"center",children:r}),(0,n.jsx)(p.E,{justify:"center",pt:"1rem",fontSize:"xl",children:x.map((function(e){return(0,n.jsxs)(a.xu,{pl:"1rem",pr:"1rem",children:[(0,n.jsx)(m.default,{href:e.url,passHref:!0,children:(0,n.jsx)(d.r,{children:e.name})}),(0,n.jsxs)("sup",{children:[" ",e.institutions.toString()]})]},e.name)}))},"authors"),(0,n.jsx)(p.E,{justify:"center",pt:"1rem",children:Object.entries(h).map((function(e){return(0,n.jsxs)(a.xu,{children:[(0,n.jsxs)("sup",{children:[e[0],"  "]}),e[1]]})}))},"institutions")]})};f.defaultProps={title:"Default Title"};var j=t(6265),b=t(980),g=t(4096);function w(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}var y=function(e){var r=(0,b.useColorMode)().colorMode;return(0,n.jsx)(g.k,function(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?w(Object(t),!0).forEach((function(r){(0,j.Z)(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):w(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}({direction:"column",alignItems:"center",justifyContent:"flex-start",bg:{light:"gray.50",dark:"gray.900"}[r],color:{light:"black",dark:"white"}[r]},e))},v=t(6034),O=t(3652),k=function(){var e=(0,b.useColorMode)(),r=e.colorMode,t=e.toggleColorMode,i="dark"===r;return(0,n.jsx)(n.Fragment,{children:(0,n.jsxs)(v.Kq,{direction:"row",position:"fixed",top:"1rem",right:"1rem",children:[(0,n.jsx)(O.lX,{htmlFor:"dark-mode-switch",mt:"-3px",opacity:"0.3",children:i?"Dark":"Light"}),(0,n.jsx)(b.Switch,{color:"green",isChecked:i,onChange:t})]})})},P=t(155),S=t(2821),C=t(1649),E="https://arxiv.org/abs/2002.00733",z="https://github.com/lukemelas/pixmatch",H=function(){return(0,n.jsxs)(v.Kq,{direction:"row",spacing:4,pt:"2rem",pb:"2rem",children:[(0,n.jsx)(m.default,{href:E,passHref:!0,children:(0,n.jsx)(P.z,{leftIcon:(0,n.jsx)(C.e3_,{size:"25px"}),colorScheme:"teal",variant:"outline",children:"Paper"})}),(0,n.jsx)(m.default,{href:z,passHref:!0,children:(0,n.jsx)(P.z,{leftIcon:(0,n.jsx)(S.idJ,{size:"25px"}),colorScheme:"teal",variant:"solid",children:"GitHub"})})]})};function _(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}var A=function(e){return(0,n.jsx)(g.k,function(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?_(Object(t),!0).forEach((function(r){(0,j.Z)(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):_(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}({as:"footer",py:"4rem"},e))},D=t(4155),I=function(){return(0,n.jsxs)(y,{children:[(0,n.jsx)(f,{title:"PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training"}),(0,n.jsx)(H,{}),(0,n.jsx)(y,{w:"90vw",h:"50.6vw",maxW:"700px",maxH:"393px",mb:"3rem",children:(0,n.jsx)("iframe",{width:"100%",height:"100%",src:"https://www.youtube.com/embed/ScMzIvxBSi4",title:"Video",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})}),(0,n.jsxs)(y,{w:"100%",maxW:"44rem",alignItems:"left",pl:"1rem",pr:"1rem",children:[(0,n.jsx)(i.X,{fontSize:"2xl",pb:"1rem",children:"Abstract"}),(0,n.jsx)(s.x,{pb:"2rem",children:"Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the idea that in order to perform well on the target domain, a model\u2019s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes."}),(0,n.jsx)(i.X,{fontSize:"2xl",pb:"1rem",children:"Approach"}),(0,n.jsx)(o.E,{src:"".concat(D.env.BASE_PATH||"","/images/diagram.jpg")}),(0,n.jsx)(s.x,{align:"center",pt:"0.5rem",pb:"0.5rem",fontSize:"small",children:"Our proposed pixelwise consistency training approach."}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Examples"}),(0,n.jsx)(o.E,{src:"".concat(D.env.BASE_PATH||"","/images/example-synthia.jpg")}),(0,n.jsx)(s.x,{align:"center",pt:"0.5rem",pb:"0.5rem",fontSize:"small",children:"Qualitative results on SYNTHIA-to-Cityscapes"}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Citation"}),(0,n.jsx)(a.xu,{w:"100%",overflow:"scroll",children:(0,n.jsxs)(c.E,{p:"0.5rem",borderRadius:"5px",w:"max-content",children:["@inproceedings{ ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0yu2021plenoctrees, ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0title={PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0author={Luke Melas-Kyriazi and Arjun K. Manrai} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0year={2021} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0booktitle={CVPR} ",(0,n.jsx)("br",{}),"}"]})}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Related Work"}),(0,n.jsxs)(l.QI,{children:[(0,n.jsx)(l.HC,{children:(0,n.jsx)(s.x,{color:"blue",children:(0,n.jsx)(m.default,{href:"#",passHref:!0,children:"First paper"})})}),(0,n.jsx)(l.HC,{children:(0,n.jsx)(s.x,{color:"blue",children:(0,n.jsx)(m.default,{href:"#",passHref:!0,children:"Second paper"})})})]})]}),(0,n.jsx)(k,{}),(0,n.jsx)(A,{children:(0,n.jsx)(s.x,{})})]})}},5301:function(e,r,t){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return t(4496)}])}},function(e){e.O(0,[774,866,617,351,736,538],(function(){return r=5301,e(e.s=r);var r}));var r=e.O();_N_E=r}]);