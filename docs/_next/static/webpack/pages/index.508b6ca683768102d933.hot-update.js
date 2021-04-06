/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("pages/index",{

/***/ "./src/pages/index.tsx":
/*!*****************************!*\
  !*** ./src/pages/index.tsx ***!
  \*****************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"./node_modules/react/jsx-dev-runtime.js\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var _chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! @chakra-ui/react */ \"./node_modules/@chakra-ui/react/dist/esm/index.js\");\n/* harmony import */ var components_Hero__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! components/Hero */ \"./src/components/Hero.tsx\");\n/* harmony import */ var components_Container__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! components/Container */ \"./src/components/Container.tsx\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/link */ \"./node_modules/next/link.js\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_3__);\n/* harmony import */ var components_DarkModeSwitch__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! components/DarkModeSwitch */ \"./src/components/DarkModeSwitch.tsx\");\n/* harmony import */ var components_LinksRow__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! components/LinksRow */ \"./src/components/LinksRow.tsx\");\n/* harmony import */ var components_Footer__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! components/Footer */ \"./src/components/Footer.tsx\");\n/* module decorator */ module = __webpack_require__.hmd(module);\n/* provided dependency */ var process = __webpack_require__(/*! process */ \"./node_modules/process/browser.js\");\n\n\nvar _jsxFileName = \"/Users/user/Documents/projects/experiments/pixmatch/src/pages/index.tsx\",\n    _this = undefined;\n\n\n\n\n\n\n\n\n\nvar Index = function Index() {\n  return /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n    children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Hero__WEBPACK_IMPORTED_MODULE_1__.Hero, {\n      title: \"PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training\"\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 13,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_LinksRow__WEBPACK_IMPORTED_MODULE_5__.LinksRow, {}, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 16,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n      w: \"90vw\",\n      h: \"50.6vw\",\n      maxW: \"700px\",\n      maxH: \"393px\",\n      mb: \"3rem\",\n      children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"iframe\", {\n        width: \"100%\",\n        height: \"100%\",\n        src: \"https://www.youtube.com/embed/ScMzIvxBSi4\",\n        title: \"Video\",\n        allow: \"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture\",\n        allowFullScreen: true\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 20,\n        columnNumber: 7\n      }, _this)\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 19,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n      w: \"100%\",\n      maxW: \"44rem\",\n      alignItems: \"left\",\n      pl: \"1rem\",\n      pr: \"1rem\",\n      children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pb: \"1rem\",\n        children: \"Abstract\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 31,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        pb: \"2rem\",\n        children: \"Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the idea that in order to perform well on the target domain, a model\\u2019s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes.\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 32,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pb: \"1rem\",\n        children: \"Approach\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 37,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"img\", {\n        src: \"\".concat(process.env.BASE_PATH || \"\", \"/images/diagram.jpg\")\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 38,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        align: \"center\",\n        pt: \"0.5rem\",\n        pb: \"0.5rem\",\n        fontSize: \"small\",\n        children: \"Our proposed pixelwise consistency training approach.\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 39,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Examples\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 42,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"img\", {\n        src: \"\".concat(process.env.BASE_PATH || \"\", \"/images/example-synthia.jpg\")\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 43,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        align: \"center\",\n        pt: \"0.5rem\",\n        pb: \"0.5rem\",\n        fontSize: \"small\",\n        children: \"Qualitative results on SYNTHIA-to-Cityscapes\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 44,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Citation\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 48,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Box, {\n        w: \"100%\",\n        overflow: \"scroll\",\n        children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Code, {\n          p: \"0.5rem\",\n          borderRadius: \"5px\",\n          w: \"max-content\",\n          children: [\"@inproceedings{ \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 52,\n            columnNumber: 32\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0yu2021plenoctrees, \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 53,\n            columnNumber: 54\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0title={PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 54,\n            columnNumber: 129\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0author={Luke Melas-Kyriazi and Arjun K. Manrai} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 55,\n            columnNumber: 93\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0year={2021} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 56,\n            columnNumber: 57\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0booktitle={CVPR} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 57,\n            columnNumber: 62\n          }, _this), \"}\"]\n        }, void 0, true, {\n          fileName: _jsxFileName,\n          lineNumber: 50,\n          columnNumber: 9\n        }, _this)\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 49,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Related Work\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 63,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.UnorderedList, {\n        children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.ListItem, {\n          children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n            color: \"gray\",\n            children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Box, {\n              color: \"gray\",\n              d: \"inline-block\",\n              children: \"(Coming Soon) \"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 67,\n              columnNumber: 13\n            }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_3___default()), {\n              href: \"#\",\n              children: \"DomainMix: Improving Domain Adaptation by Adversarial Self-Training with Mixed Source and Target Data\"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 68,\n              columnNumber: 13\n            }, _this)]\n          }, void 0, true, {\n            fileName: _jsxFileName,\n            lineNumber: 66,\n            columnNumber: 11\n          }, _this)\n        }, void 0, false, {\n          fileName: _jsxFileName,\n          lineNumber: 65,\n          columnNumber: 9\n        }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.ListItem, {\n          children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n            color: \"blue\",\n            children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_3___default()), {\n              href: \"https://github.com/ZJULearning/MaxSquareLoss\",\n              children: \"Domain Adaptation for Semantic Segmentation with Maximum Squares Loss\"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 75,\n              columnNumber: 13\n            }, _this)\n          }, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 74,\n            columnNumber: 11\n          }, _this)\n        }, void 0, false, {\n          fileName: _jsxFileName,\n          lineNumber: 73,\n          columnNumber: 9\n        }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.ListItem, {\n          children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n            color: \"blue\",\n            children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_3___default()), {\n              href: \"https://github.com/valeoai/ADVENT\",\n              children: \"ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation\"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 82,\n              columnNumber: 13\n            }, _this)\n          }, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 81,\n            columnNumber: 11\n          }, _this)\n        }, void 0, false, {\n          fileName: _jsxFileName,\n          lineNumber: 80,\n          columnNumber: 9\n        }, _this)]\n      }, void 0, true, {\n        fileName: _jsxFileName,\n        lineNumber: 64,\n        columnNumber: 7\n      }, _this)]\n    }, void 0, true, {\n      fileName: _jsxFileName,\n      lineNumber: 28,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_DarkModeSwitch__WEBPACK_IMPORTED_MODULE_4__.DarkModeSwitch, {}, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 96,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Footer__WEBPACK_IMPORTED_MODULE_6__.Footer, {\n      children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {}, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 98,\n        columnNumber: 7\n      }, _this)\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 97,\n      columnNumber: 5\n    }, _this)]\n  }, void 0, true, {\n    fileName: _jsxFileName,\n    lineNumber: 10,\n    columnNumber: 3\n  }, _this);\n};\n\n_c = Index;\n/* harmony default export */ __webpack_exports__[\"default\"] = (Index);\n\nvar _c;\n\n$RefreshReg$(_c, \"Index\");\n\n;\n    var _a, _b;\n    // Legacy CSS implementations will `eval` browser code in a Node.js context\n    // to extract CSS. For backwards compatibility, we need to check we're in a\n    // browser context before continuing.\n    if (typeof self !== 'undefined' &&\n        // AMP / No-JS mode does not inject these helpers:\n        '$RefreshHelpers$' in self) {\n        var currentExports = module.__proto__.exports;\n        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;\n        // This cannot happen in MainTemplate because the exports mismatch between\n        // templating and execution.\n        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n        // A module can be accepted automatically based on its exports, e.g. when\n        // it is a Refresh Boundary.\n        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n            // Save the previous exports on update so we can compare the boundary\n            // signatures.\n            module.hot.dispose(function (data) {\n                data.prevExports = currentExports;\n            });\n            // Unconditionally accept an update to this module, we'll check if it's\n            // still a Refresh Boundary later.\n            module.hot.accept();\n            // This field is set when the previous version of this module was a\n            // Refresh Boundary, letting us know we need to check for invalidation or\n            // enqueue an update.\n            if (prevExports !== null) {\n                // A boundary can become ineligible if its exports are incompatible\n                // with the previous exports.\n                //\n                // For example, if you add/remove/change exports, we'll want to\n                // re-execute the importing modules, and force those components to\n                // re-render. Similarly, if you convert a class component to a\n                // function, we want to invalidate the boundary.\n                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {\n                    module.hot.invalidate();\n                }\n                else {\n                    self.$RefreshHelpers$.scheduleUpdate();\n                }\n            }\n        }\n        else {\n            // Since we just executed the code for the module, it's possible that the\n            // new exports made it ineligible for being a boundary.\n            // We only care about the case when we were _previously_ a boundary,\n            // because we already accepted this update (accidental side effect).\n            var isNoLongerABoundary = prevExports !== null;\n            if (isNoLongerABoundary) {\n                module.hot.invalidate();\n            }\n        }\n    }\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3JjL3BhZ2VzL2luZGV4LnRzeD80MWUwIl0sIm5hbWVzIjpbIkluZGV4IiwicHJvY2VzcyIsImVudiIsIkJBU0VfUEFUSCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUEsSUFBTUEsS0FBSyxHQUFHLFNBQVJBLEtBQVE7QUFBQSxzQkFDWiw4REFBQywyREFBRDtBQUFBLDRCQUdFLDhEQUFDLGlEQUFEO0FBQU0sV0FBSyxFQUFDO0FBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQUhGLGVBTUUsOERBQUMseURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQU5GLGVBU0UsOERBQUMsMkRBQUQ7QUFBVyxPQUFDLEVBQUMsTUFBYjtBQUFvQixPQUFDLEVBQUMsUUFBdEI7QUFBK0IsVUFBSSxFQUFDLE9BQXBDO0FBQTRDLFVBQUksRUFBQyxPQUFqRDtBQUF5RCxRQUFFLEVBQUMsTUFBNUQ7QUFBQSw2QkFDRTtBQUNFLGFBQUssRUFBQyxNQURSO0FBQ2UsY0FBTSxFQUFDLE1BRHRCO0FBRUUsV0FBRyxFQUFDLDJDQUZOO0FBR0UsYUFBSyxFQUFDLE9BSFI7QUFJRSxhQUFLLEVBQUMsMEZBSlI7QUFJbUcsdUJBQWU7QUFKbEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQURGO0FBQUE7QUFBQTtBQUFBO0FBQUEsYUFURixlQWtCRSw4REFBQywyREFBRDtBQUFXLE9BQUMsRUFBQyxNQUFiO0FBQW9CLFVBQUksRUFBQyxPQUF6QjtBQUFpQyxnQkFBVSxFQUFDLE1BQTVDO0FBQW1ELFFBQUUsRUFBQyxNQUF0RDtBQUE2RCxRQUFFLEVBQUMsTUFBaEU7QUFBQSw4QkFHRSw4REFBQyxxREFBRDtBQUFTLGdCQUFRLEVBQUMsS0FBbEI7QUFBd0IsVUFBRSxFQUFDLE1BQTNCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGVBSEYsZUFJRSw4REFBQyxrREFBRDtB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QUFmO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBREY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQURGO0FBQUE7QUFBQTtBQUFBO0FBQUEsaUJBVEYsZUFnQkUsOERBQUMsc0RBQUQ7QUFBQSxpQ0FDRSw4REFBQyxrREFBRDtBQUFNLGlCQUFLLEVBQUMsTUFBWjtBQUFBLG1DQUNFLDhEQUFDLGtEQUFEO0FBQVUsa0JBQUksRUFBQyxtQ0FBZjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQURGO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFERjtBQUFBO0FBQUE7QUFBQTtBQUFBLGlCQWhCRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsZUFwQ0Y7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGFBbEJGLGVBc0ZFLDhEQUFDLHFFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUEsYUF0RkYsZUF1RkUsOERBQUMscURBQUQ7QUFBQSw2QkFDRSw4REFBQyxrREFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBREY7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQXZGRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FEWTtBQUFBLENBQWQ7O0tBQU1ILEs7QUE4Rk4sK0RBQWVBLEtBQWYiLCJmaWxlIjoiLi9zcmMvcGFnZXMvaW5kZXgudHN4LmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgTGluayBhcyBDaGFrcmFMaW5rLCBJbWFnZSBhcyBDaGFrcmFJbWFnZSwgVGV4dCwgQ29kZSwgTGlzdEl0ZW0sIEhlYWRpbmcsIFVub3JkZXJlZExpc3QsIEJveCwgfSBmcm9tICdAY2hha3JhLXVpL3JlYWN0J1xuaW1wb3J0IHsgSGVybyB9IGZyb20gJ2NvbXBvbmVudHMvSGVybydcbmltcG9ydCB7IENvbnRhaW5lciB9IGZyb20gJ2NvbXBvbmVudHMvQ29udGFpbmVyJ1xuaW1wb3J0IE5leHRMaW5rIGZyb20gJ25leHQvbGluaydcbmltcG9ydCB7IERhcmtNb2RlU3dpdGNoIH0gZnJvbSAnY29tcG9uZW50cy9EYXJrTW9kZVN3aXRjaCdcbmltcG9ydCB7IExpbmtzUm93IH0gZnJvbSAnY29tcG9uZW50cy9MaW5rc1JvdydcbmltcG9ydCB7IEZvb3RlciB9IGZyb20gJ2NvbXBvbmVudHMvRm9vdGVyJ1xuXG5jb25zdCBJbmRleCA9ICgpID0+IChcbiAgPENvbnRhaW5lcj5cblxuICAgIHsvKiBFZGl0IGF1dGhvciBpbmZvICovfVxuICAgIDxIZXJvIHRpdGxlPVwiUGl4TWF0Y2g6IFVuc3VwZXJ2aXNlZCBEb21haW4gQWRhcHRhdGlvbiB2aWEgUGl4ZWx3aXNlIENvbnNpc3RlbmN5IFRyYWluaW5nXCIgLz5cblxuICAgIHsvKiBUT0RPOiBBZGQgcGFwZXIvZ2l0aHViIGxpbmtzIGhlcmUgKi99XG4gICAgPExpbmtzUm93IC8+XG5cbiAgICB7LyogVE9ETzogQWRkIHZpZGVvICovfVxuICAgIDxDb250YWluZXIgdz1cIjkwdndcIiBoPVwiNTAuNnZ3XCIgbWF4Vz1cIjcwMHB4XCIgbWF4SD1cIjM5M3B4XCIgbWI9XCIzcmVtXCI+XG4gICAgICA8aWZyYW1lXG4gICAgICAgIHdpZHRoPVwiMTAwJVwiIGhlaWdodD1cIjEwMCVcIlxuICAgICAgICBzcmM9XCJodHRwczovL3d3dy55b3V0dWJlLmNvbS9lbWJlZC9TY016SXZ4QlNpNFwiXG4gICAgICAgIHRpdGxlPVwiVmlkZW9cIlxuICAgICAgICBhbGxvdz1cImFjY2VsZXJvbWV0ZXI7IGF1dG9wbGF5OyBjbGlwYm9hcmQtd3JpdGU7IGVuY3J5cHRlZC1tZWRpYTsgZ3lyb3Njb3BlOyBwaWN0dXJlLWluLXBpY3R1cmVcIiBhbGxvd0Z1bGxTY3JlZW4+XG4gICAgICA8L2lmcmFtZT5cbiAgICA8L0NvbnRhaW5lcj5cblxuICAgIDxDb250YWluZXIgdz1cIjEwMCVcIiBtYXhXPVwiNDRyZW1cIiBhbGlnbkl0ZW1zPVwibGVmdFwiIHBsPVwiMXJlbVwiIHByPVwiMXJlbVwiPlxuXG4gICAgICB7LyogQWJzdHJhY3QgKi99XG4gICAgICA8SGVhZGluZyBmb250U2l6ZT1cIjJ4bFwiIHBiPVwiMXJlbVwiPkFic3RyYWN0PC9IZWFkaW5nPlxuICAgICAgPFRleHQgcGI9XCIycmVtXCI+XG4gICAgICAgIFVuc3VwZXJ2aXNlZCBkb21haW4gYWRhcHRhdGlvbiBpcyBhIHByb21pc2luZyB0ZWNobmlxdWUgZm9yIHNlbWFudGljIHNlZ21lbnRhdGlvbiBhbmQgb3RoZXIgY29tcHV0ZXIgdmlzaW9uIHRhc2tzIGZvciB3aGljaCBsYXJnZS1zY2FsZSBkYXRhIGFubm90YXRpb24gaXMgY29zdGx5IGFuZCB0aW1lLWNvbnN1bWluZy4gSW4gc2VtYW50aWMgc2VnbWVudGF0aW9uLCBpdCBpcyBhdHRyYWN0aXZlIHRvIHRyYWluIG1vZGVscyBvbiBhbm5vdGF0ZWQgaW1hZ2VzIGZyb20gYSBzaW11bGF0ZWQgKHNvdXJjZSkgZG9tYWluIGFuZCBkZXBsb3kgdGhlbSBvbiByZWFsICh0YXJnZXQpIGRvbWFpbnMuIEluIHRoaXMgd29yaywgd2UgcHJlc2VudCBhIG5vdmVsIGZyYW1ld29yayBmb3IgdW5zdXBlcnZpc2VkIGRvbWFpbiBhZGFwdGF0aW9uIGJhc2VkIG9uIHRoZSBub3Rpb24gb2YgdGFyZ2V0LWRvbWFpbiBjb25zaXN0ZW5jeSB0cmFpbmluZy4gSW50dWl0aXZlbHksIG91ciB3b3JrIGlzIGJhc2VkIG9uIHRoZSBpZGVhIHRoYXQgaW4gb3JkZXIgdG8gcGVyZm9ybSB3ZWxsIG9uIHRoZSB0YXJnZXQgZG9tYWluLCBhIG1vZGVs4oCZcyBvdXRwdXQgc2hvdWxkIGJlIGNvbnNpc3RlbnQgd2l0aCByZXNwZWN0IHRvIHNtYWxsIHBlcnR1cmJhdGlvbnMgb2YgaW5wdXRzIGluIHRoZSB0YXJnZXQgZG9tYWluLiBTcGVjaWZpY2FsbHksIHdlIGludHJvZHVjZSBhIG5ldyBsb3NzIHRlcm0gdG8gZW5mb3JjZSBwaXhlbHdpc2UgY29uc2lzdGVuY3kgYmV0d2VlbiB0aGUgbW9kZWwncyBwcmVkaWN0aW9ucyBvbiBhIHRhcmdldCBpbWFnZSBhbmQgYSBwZXJ0dXJiZWQgdmVyc2lvbiBvZiB0aGUgc2FtZSBpbWFnZS4gSW4gY29tcGFyaXNvbiB0byBwb3B1bGFyIGFkdmVyc2FyaWFsIGFkYXB0YXRpb24gbWV0aG9kcywgb3VyIGFwcHJvYWNoIGlzIHNpbXBsZXIsIGVhc2llciB0byBpbXBsZW1lbnQsIGFuZCBtb3JlIG1lbW9yeS1lZmZpY2llbnQgZHVyaW5nIHRyYWluaW5nLiBFeHBlcmltZW50cyBhbmQgZXh0ZW5zaXZlIGFibGF0aW9uIHN0dWRpZXMgZGVtb25zdHJhdGUgdGhhdCBvdXIgc2ltcGxlIGFwcHJvYWNoIGFjaGlldmVzIHJlbWFya2FibHkgc3Ryb25nIHJlc3VsdHMgb24gdHdvIGNoYWxsZW5naW5nIHN5bnRoZXRpYy10by1yZWFsIGJlbmNobWFya3MsIEdUQTUtdG8tQ2l0eXNjYXBlcyBhbmQgU1lOVEhJQS10by1DaXR5c2NhcGVzLlxuICAgICAgPC9UZXh0PlxuXG4gICAgICB7LyogRXhhbXBsZSAqL31cbiAgICAgIDxIZWFkaW5nIGZvbnRTaXplPVwiMnhsXCIgcGI9XCIxcmVtXCI+QXBwcm9hY2g8L0hlYWRpbmc+XG4gICAgICA8aW1nIHNyYz17YCR7cHJvY2Vzcy5lbnYuQkFTRV9QQVRIIHx8IFwiXCJ9L2ltYWdlcy9kaWFncmFtLmpwZ2B9IC8+XG4gICAgICA8VGV4dCBhbGlnbj1cImNlbnRlclwiIHB0PVwiMC41cmVtXCIgcGI9XCIwLjVyZW1cIiBmb250U2l6ZT1cInNtYWxsXCI+T3VyIHByb3Bvc2VkIHBpeGVsd2lzZSBjb25zaXN0ZW5jeSB0cmFpbmluZyBhcHByb2FjaC48L1RleHQ+XG5cbiAgICAgIHsvKiBBbm90aGVyIFNlY3Rpb24gKi99XG4gICAgICA8SGVhZGluZyBmb250U2l6ZT1cIjJ4bFwiIHB0PVwiMnJlbVwiIHBiPVwiMXJlbVwiPkV4YW1wbGVzPC9IZWFkaW5nPlxuICAgICAgPGltZyBzcmM9e2Ake3Byb2Nlc3MuZW52LkJBU0VfUEFUSCB8fCBcIlwifS9pbWFnZXMvZXhhbXBsZS1zeW50aGlhLmpwZ2B9IC8+XG4gICAgICA8VGV4dCBhbGlnbj1cImNlbnRlclwiIHB0PVwiMC41cmVtXCIgcGI9XCIwLjVyZW1cIiBmb250U2l6ZT1cInNtYWxsXCI+UXVhbGl0YXRpdmUgcmVzdWx0cyBvbiBTWU5USElBLXRvLUNpdHlzY2FwZXM8L1RleHQ+XG5cblxuICAgICAgey8qIENpdGF0aW9uICovfVxuICAgICAgPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwdD1cIjJyZW1cIiBwYj1cIjFyZW1cIj5DaXRhdGlvbjwvSGVhZGluZz5cbiAgICAgIDxCb3ggdz1cIjEwMCVcIiBvdmVyZmxvdz1cInNjcm9sbFwiPlxuICAgICAgICA8Q29kZSBwPVwiMC41cmVtXCIgYm9yZGVyUmFkaXVzPVwiNXB4XCIgdz1cIm1heC1jb250ZW50XCI+XG4gICAgICAgICAgey8qIHc9XCIxNTAlXCI+ICovfVxuICAgICAgICAgIEBpbnByb2NlZWRpbmdzJiMxMjM7IDxiciAvPlxuICAgICAgICAgICZuYnNwOyZuYnNwOyZuYnNwOyZuYnNwO3l1MjAyMXBsZW5vY3RyZWVzLCA8YnIgLz5cbiAgICAgICAgICAmbmJzcDsmbmJzcDsmbmJzcDsmbmJzcDt0aXRsZT0mIzEyMztQaXhNYXRjaDogVW5zdXBlcnZpc2VkIERvbWFpbiBBZGFwdGF0aW9uIHZpYSBQaXhlbHdpc2UgQ29uc2lzdGVuY3kgVHJhaW5pbmcmIzEyNTsgPGJyIC8+XG4gICAgICAgICAgJm5ic3A7Jm5ic3A7Jm5ic3A7Jm5ic3A7YXV0aG9yPSYjMTIzO0x1a2UgTWVsYXMtS3lyaWF6aSBhbmQgQXJqdW4gSy4gTWFucmFpJiMxMjU7IDxiciAvPlxuICAgICAgICAgICZuYnNwOyZuYnNwOyZuYnNwOyZuYnNwO3llYXI9JiMxMjM7MjAyMSYjMTI1OyA8YnIgLz5cbiAgICAgICAgICAmbmJzcDsmbmJzcDsmbmJzcDsmbmJzcDtib29rdGl0bGU9JiMxMjM7Q1ZQUiYjMTI1OyA8YnIgLz5cbiAgICAgICYjMTI1O1xuICAgICAgPC9Db2RlPlxuICAgICAgPC9Cb3g+XG5cbiAgICAgIHsvKiBSZWxhdGVkIFdvcmsgKi99XG4gICAgICA8SGVhZGluZyBmb250U2l6ZT1cIjJ4bFwiIHB0PVwiMnJlbVwiIHBiPVwiMXJlbVwiPlJlbGF0ZWQgV29yazwvSGVhZGluZz5cbiAgICAgIDxVbm9yZGVyZWRMaXN0PlxuICAgICAgICA8TGlzdEl0ZW0+XG4gICAgICAgICAgPFRleHQgY29sb3I9XCJncmF5XCI+XG4gICAgICAgICAgICA8Qm94IGNvbG9yPVwiZ3JheVwiIGQ9XCJpbmxpbmUtYmxvY2tcIj4oQ29taW5nIFNvb24pIDwvQm94PlxuICAgICAgICAgICAgPE5leHRMaW5rIGhyZWY9XCIjXCI+XG4gICAgICAgICAgICAgIERvbWFpbk1peDogSW1wcm92aW5nIERvbWFpbiBBZGFwdGF0aW9uIGJ5IEFkdmVyc2FyaWFsIFNlbGYtVHJhaW5pbmcgd2l0aCBNaXhlZCBTb3VyY2UgYW5kIFRhcmdldCBEYXRhXG4gICAgICAgICAgICA8L05leHRMaW5rPlxuICAgICAgICAgIDwvVGV4dD5cbiAgICAgICAgPC9MaXN0SXRlbT5cbiAgICAgICAgPExpc3RJdGVtPlxuICAgICAgICAgIDxUZXh0IGNvbG9yPVwiYmx1ZVwiPlxuICAgICAgICAgICAgPE5leHRMaW5rIGhyZWY9XCJodHRwczovL2dpdGh1Yi5jb20vWkpVTGVhcm5pbmcvTWF4U3F1YXJlTG9zc1wiPlxuICAgICAgICAgICAgICBEb21haW4gQWRhcHRhdGlvbiBmb3IgU2VtYW50aWMgU2VnbWVudGF0aW9uIHdpdGggTWF4aW11bSBTcXVhcmVzIExvc3NcbiAgICAgICAgICAgIDwvTmV4dExpbms+XG4gICAgICAgICAgPC9UZXh0PlxuICAgICAgICA8L0xpc3RJdGVtPlxuICAgICAgICA8TGlzdEl0ZW0+XG4gICAgICAgICAgPFRleHQgY29sb3I9XCJibHVlXCI+XG4gICAgICAgICAgICA8TmV4dExpbmsgaHJlZj1cImh0dHBzOi8vZ2l0aHViLmNvbS92YWxlb2FpL0FEVkVOVFwiPlxuICAgICAgICAgICAgICBBRFZFTlQ6IEFkdmVyc2FyaWFsIEVudHJvcHkgTWluaW1pemF0aW9uIGZvciBEb21haW4gQWRhcHRhdGlvbiBpbiBTZW1hbnRpYyBTZWdtZW50YXRpb25cbiAgICAgICAgICAgIDwvTmV4dExpbms+XG4gICAgICAgICAgPC9UZXh0PlxuICAgICAgICA8L0xpc3RJdGVtPlxuICAgICAgPC9Vbm9yZGVyZWRMaXN0PlxuXG4gICAgICB7LyogQWNrbm93bGVkZ2VtZW50cyAqL31cbiAgICAgIHsvKiA8SGVhZGluZyBmb250U2l6ZT1cIjJ4bFwiIHB0PVwiMnJlbVwiIHBiPVwiMXJlbVwiPkFja25vd2xlZGdlbWVudHM8L0hlYWRpbmc+XG4gICAgICA8VGV4dCA+XG4gICAgICAgIFdlIHRoYW5rIHh5eiBmb3IgYWJjLi4uXG4gICAgICA8L1RleHQ+ICovfVxuICAgIDwvQ29udGFpbmVyPlxuXG4gICAgPERhcmtNb2RlU3dpdGNoIC8+XG4gICAgPEZvb3Rlcj5cbiAgICAgIDxUZXh0PjwvVGV4dD5cbiAgICA8L0Zvb3Rlcj5cbiAgPC9Db250YWluZXIgPlxuKVxuXG5leHBvcnQgZGVmYXVsdCBJbmRleFxuIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./src/pages/index.tsx\n");

/***/ })

});