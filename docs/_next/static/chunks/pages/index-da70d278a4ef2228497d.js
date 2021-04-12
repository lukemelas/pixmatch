(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{4496:function(e,t,r){"use strict";r.r(t),r.d(t,{default:function(){return T}});var n=r(5893),i=r(336),o=r(4115),a=r(8017),s=r(7922),c=r(296),l=r(3719),d=r(9444),p=r(7544),m=r(1664),u={1:"Harvard University",2:"Oxford University",3:"Boston Children's Hospital"},h=[{name:"Luke Melas-Kyriazi",institutions:[1,2],url:"https://lukemelas.github.io/"},{name:"Arjun K. Manrai",institutions:[1,3],url:"https://www.childrenshospital.org/research/researchers/m/arjun-manrai"}],x=function(e){var t=e.title;return(0,n.jsxs)(p.Container,{children:[(0,n.jsx)(i.X,{fontSize:"2xl",pt:"3rem",maxW:"42rem",textAlign:"center",children:t}),(0,n.jsx)(l.E,{justify:"center",pt:"1rem",fontSize:"xl",children:h.map((function(e){return(0,n.jsxs)(a.xu,{pl:"1rem",pr:"1rem",children:[(0,n.jsx)(m.default,{href:e.url,passHref:!0,children:(0,n.jsx)(d.r,{children:e.name})}),(0,n.jsxs)("sup",{children:[" ",e.institutions.toString()]})]},e.name)}))},"authors"),(0,n.jsx)(l.E,{justify:"center",pt:"1rem",children:Object.entries(u).map((function(e){return(0,n.jsxs)(a.xu,{children:[(0,n.jsxs)("sup",{children:[e[0],"  "]}),e[1]]})}))},"institutions")]})};x.defaultProps={title:"Default Title"};var f=r(6265),j=r(980),g=r(4096);function b(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}var y=function(e){var t=(0,j.useColorMode)().colorMode;return(0,n.jsx)(g.k,function(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?b(Object(r),!0).forEach((function(t){(0,f.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):b(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}({direction:"column",alignItems:"center",justifyContent:"flex-start",bg:{light:"gray.50",dark:"gray.900"}[t],color:{light:"black",dark:"white"}[t]},e))},w=r(6034),v=r(3652),O=function(){var e=(0,j.useColorMode)(),t=e.colorMode,r=e.toggleColorMode,i="dark"===t;return(0,n.jsx)(n.Fragment,{children:(0,n.jsxs)(w.Kq,{direction:"row",position:"fixed",top:"1rem",right:"1rem",children:[(0,n.jsx)(v.lX,{htmlFor:"dark-mode-switch",mt:"-3px",opacity:"0.3",children:i?"Dark":"Light"}),(0,n.jsx)(j.Switch,{color:"green",isChecked:i,onChange:r})]})})},S=r(155),k=r(2821),P=r(1649),C="https://arxiv.org/abs/2002.00733",A="https://github.com/lukemelas/pixmatch",D=function(){return(0,n.jsxs)(w.Kq,{direction:"row",spacing:4,pt:"2rem",pb:"2rem",children:[(0,n.jsx)(m.default,{href:C,passHref:!0,children:(0,n.jsx)(S.z,{leftIcon:(0,n.jsx)(P.e3_,{size:"25px"}),colorScheme:"teal",variant:"outline",children:"Paper"})}),(0,n.jsx)(m.default,{href:A,passHref:!0,children:(0,n.jsx)(S.z,{leftIcon:(0,n.jsx)(k.idJ,{size:"25px"}),colorScheme:"teal",variant:"solid",children:"GitHub"})})]})};function E(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}var M=function(e){return(0,n.jsx)(g.k,function(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?E(Object(r),!0).forEach((function(t){(0,f.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):E(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}({as:"footer",py:"4rem"},e))},z=r(4155),T=function(){return(0,n.jsxs)(y,{children:[(0,n.jsx)(x,{title:"PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training"}),(0,n.jsx)(D,{}),(0,n.jsx)(y,{w:"90vw",h:"50.6vw",maxW:"700px",maxH:"393px",mb:"3rem",children:(0,n.jsx)("iframe",{width:"100%",height:"100%",src:"https://www.youtube.com/embed/ScMzIvxBSi4",title:"Video",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})}),(0,n.jsxs)(y,{w:"100%",maxW:"44rem",alignItems:"left",pl:"1rem",pr:"1rem",children:[(0,n.jsx)(i.X,{fontSize:"2xl",pb:"1rem",children:"Abstract"}),(0,n.jsx)(o.x,{pb:"2rem",children:"Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the idea that in order to perform well on the target domain, a model\u2019s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes."}),(0,n.jsx)(i.X,{fontSize:"2xl",pb:"1rem",children:"Approach"}),(0,n.jsx)("img",{src:"".concat(z.env.BASE_PATH||"","/images/diagram.jpg")}),(0,n.jsx)(o.x,{align:"center",pt:"0.5rem",pb:"0.5rem",fontSize:"small",children:"Our proposed unsupervised domain adaptation approach for semantic segmentation, PixMatch, which employs consistency training and pseudolabeling to enforce consistency on the target domain."}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Examples"}),(0,n.jsx)("img",{src:"".concat(z.env.BASE_PATH||"","/images/example-synthia.jpg")}),(0,n.jsx)(o.x,{align:"center",pt:"0.5rem",pb:"0.5rem",fontSize:"small",children:"Qualitative examples of our consistency training method and prior methods on SYNTHIA-to-Cityscapes. The final column shows our baseline model with augmentation-based perturbations. Note that these images are not hand-picked; they are the first 5 images in the Cityscapes validation set."}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Citation"}),(0,n.jsx)(a.xu,{w:"100%",overflow:"scroll",children:(0,n.jsxs)(s.E,{p:"0.5rem",borderRadius:"5px",w:"max-content",children:["@inproceedings{ ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0yu2021plenoctrees, ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0title={PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0author={Luke Melas-Kyriazi and Arjun K. Manrai} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0year={2021} ",(0,n.jsx)("br",{}),"\xa0\xa0\xa0\xa0booktitle={CVPR} ",(0,n.jsx)("br",{}),"}"]})}),(0,n.jsx)(i.X,{fontSize:"2xl",pt:"2rem",pb:"1rem",children:"Related Work"}),(0,n.jsxs)(c.QI,{children:[(0,n.jsx)(c.HC,{children:(0,n.jsxs)(o.x,{color:"gray",children:[(0,n.jsx)(a.xu,{color:"black",d:"inline-block",pr:"0.5rem",children:"(Coming Soon)"}),(0,n.jsx)(m.default,{href:"#",children:"DomainMix: Improving Domain Adaptation by Adversarial Self-Training with Mixed Source and Target Data"})]})}),(0,n.jsx)(c.HC,{children:(0,n.jsx)(o.x,{color:"blue",children:(0,n.jsx)(m.default,{href:"https://github.com/ZJULearning/MaxSquareLoss",children:"Domain Adaptation for Semantic Segmentation with Maximum Squares Loss"})})}),(0,n.jsx)(c.HC,{children:(0,n.jsx)(o.x,{color:"blue",children:(0,n.jsx)(m.default,{href:"https://github.com/valeoai/ADVENT",children:"ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation"})})})]})]}),(0,n.jsx)(O,{}),(0,n.jsx)(M,{children:(0,n.jsx)(o.x,{})})]})}},5301:function(e,t,r){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return r(4496)}])}},function(e){e.O(0,[774,866,617,351,736,75],(function(){return t=5301,e(e.s=t);var t}));var t=e.O();_N_E=t}]);