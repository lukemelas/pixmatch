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
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"./node_modules/react/jsx-dev-runtime.js\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var _chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! @chakra-ui/react */ \"./node_modules/@chakra-ui/react/dist/esm/index.js\");\n/* harmony import */ var components_Hero__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! components/Hero */ \"./src/components/Hero.tsx\");\n/* harmony import */ var components_Container__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! components/Container */ \"./src/components/Container.tsx\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/link */ \"./node_modules/next/link.js\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_3__);\n/* harmony import */ var components_DarkModeSwitch__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! components/DarkModeSwitch */ \"./src/components/DarkModeSwitch.tsx\");\n/* harmony import */ var components_LinksRow__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! components/LinksRow */ \"./src/components/LinksRow.tsx\");\n/* harmony import */ var components_Footer__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! components/Footer */ \"./src/components/Footer.tsx\");\n/* module decorator */ module = __webpack_require__.hmd(module);\n/* provided dependency */ var process = __webpack_require__(/*! process */ \"./node_modules/process/browser.js\");\n\n\nvar _jsxFileName = \"/Users/user/Documents/projects/experiments/pixmatch/src/pages/index.tsx\",\n    _this = undefined;\n\n\n\n\n\n\n\n\n\nvar Index = function Index() {\n  return /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n    children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Hero__WEBPACK_IMPORTED_MODULE_1__.Hero, {\n      title: \"PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training\"\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 13,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_LinksRow__WEBPACK_IMPORTED_MODULE_5__.LinksRow, {}, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 16,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n      w: \"90vw\",\n      h: \"50.6vw\",\n      maxW: \"700px\",\n      maxH: \"393px\",\n      mb: \"3rem\",\n      children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"iframe\", {\n        width: \"100%\",\n        height: \"100%\",\n        src: \"https://www.youtube.com/embed/ScMzIvxBSi4\",\n        title: \"Video\",\n        allow: \"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture\",\n        allowFullScreen: true\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 20,\n        columnNumber: 7\n      }, _this)\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 19,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Container__WEBPACK_IMPORTED_MODULE_2__.Container, {\n      w: \"100%\",\n      maxW: \"44rem\",\n      alignItems: \"left\",\n      pl: \"1rem\",\n      pr: \"1rem\",\n      children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pb: \"1rem\",\n        children: \"Abstract\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 31,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        pb: \"2rem\",\n        children: \"Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the idea that in order to perform well on the target domain, a model\\u2019s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and a perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes.\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 32,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pb: \"1rem\",\n        children: \"Approach\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 37,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Image, {\n        src: \"\".concat(process.env.BASE_PATH || \"\", \"/images/diagram.jpg\")\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 38,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        align: \"center\",\n        pt: \"0.5rem\",\n        pb: \"0.5rem\",\n        fontSize: \"small\",\n        children: \"Our proposed pixelwise consistency training approach.\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 39,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Examples\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 42,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Image, {\n        src: \"\".concat(process.env.BASE_PATH || \"\", \"/images/example-synthia.jpg\")\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 43,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n        align: \"center\",\n        pt: \"0.5rem\",\n        pb: \"0.5rem\",\n        fontSize: \"small\",\n        children: \"Qualitative results on SYNTHIA-to-Cityscapes\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 44,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Citation\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 48,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Box, {\n        w: \"100%\",\n        overflow: \"scroll\",\n        children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Code, {\n          p: \"0.5rem\",\n          borderRadius: \"5px\",\n          children: [\"@inproceedings{ \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 52,\n            columnNumber: 32\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0yu2021plenoctrees, \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 53,\n            columnNumber: 54\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0title={PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 54,\n            columnNumber: 129\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0author={Luke Melas-Kyriazi and Arjun K. Manrai} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 55,\n            columnNumber: 93\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0year={2021} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 56,\n            columnNumber: 57\n          }, _this), \"\\xA0\\xA0\\xA0\\xA0booktitle={CVPR} \", /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"br\", {}, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 57,\n            columnNumber: 62\n          }, _this), \"}\"]\n        }, void 0, true, {\n          fileName: _jsxFileName,\n          lineNumber: 50,\n          columnNumber: 9\n        }, _this)\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 49,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Heading, {\n        fontSize: \"2xl\",\n        pt: \"2rem\",\n        pb: \"1rem\",\n        children: \"Related Work\"\n      }, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 63,\n        columnNumber: 7\n      }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.UnorderedList, {\n        children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.ListItem, {\n          children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n            color: \"blue\",\n            children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_3___default()), {\n              href: \"#\",\n              passHref: true,\n              children: \"First paper\"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 67,\n              columnNumber: 13\n            }, _this)\n          }, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 66,\n            columnNumber: 11\n          }, _this)\n        }, void 0, false, {\n          fileName: _jsxFileName,\n          lineNumber: 65,\n          columnNumber: 9\n        }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.ListItem, {\n          children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {\n            color: \"blue\",\n            children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_3___default()), {\n              href: \"#\",\n              passHref: true,\n              children: \"Second paper\"\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 74,\n              columnNumber: 13\n            }, _this)\n          }, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 73,\n            columnNumber: 11\n          }, _this)\n        }, void 0, false, {\n          fileName: _jsxFileName,\n          lineNumber: 72,\n          columnNumber: 9\n        }, _this)]\n      }, void 0, true, {\n        fileName: _jsxFileName,\n        lineNumber: 64,\n        columnNumber: 7\n      }, _this)]\n    }, void 0, true, {\n      fileName: _jsxFileName,\n      lineNumber: 28,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_DarkModeSwitch__WEBPACK_IMPORTED_MODULE_4__.DarkModeSwitch, {}, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 88,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(components_Footer__WEBPACK_IMPORTED_MODULE_6__.Footer, {\n      children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_7__.Text, {}, void 0, false, {\n        fileName: _jsxFileName,\n        lineNumber: 90,\n        columnNumber: 7\n      }, _this)\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 89,\n      columnNumber: 5\n    }, _this)]\n  }, void 0, true, {\n    fileName: _jsxFileName,\n    lineNumber: 10,\n    columnNumber: 3\n  }, _this);\n};\n\n_c = Index;\n/* harmony default export */ __webpack_exports__[\"default\"] = (Index);\n\nvar _c;\n\n$RefreshReg$(_c, \"Index\");\n\n;\n    var _a, _b;\n    // Legacy CSS implementations will `eval` browser code in a Node.js context\n    // to extract CSS. For backwards compatibility, we need to check we're in a\n    // browser context before continuing.\n    if (typeof self !== 'undefined' &&\n        // AMP / No-JS mode does not inject these helpers:\n        '$RefreshHelpers$' in self) {\n        var currentExports = module.__proto__.exports;\n        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;\n        // This cannot happen in MainTemplate because the exports mismatch between\n        // templating and execution.\n        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n        // A module can be accepted automatically based on its exports, e.g. when\n        // it is a Refresh Boundary.\n        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n            // Save the previous exports on update so we can compare the boundary\n            // signatures.\n            module.hot.dispose(function (data) {\n                data.prevExports = currentExports;\n            });\n            // Unconditionally accept an update to this module, we'll check if it's\n            // still a Refresh Boundary later.\n            module.hot.accept();\n            // This field is set when the previous version of this module was a\n            // Refresh Boundary, letting us know we need to check for invalidation or\n            // enqueue an update.\n            if (prevExports !== null) {\n                // A boundary can become ineligible if its exports are incompatible\n                // with the previous exports.\n                //\n                // For example, if you add/remove/change exports, we'll want to\n                // re-execute the importing modules, and force those components to\n                // re-render. Similarly, if you convert a class component to a\n                // function, we want to invalidate the boundary.\n                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {\n                    module.hot.invalidate();\n                }\n                else {\n                    self.$RefreshHelpers$.scheduleUpdate();\n                }\n            }\n        }\n        else {\n            // Since we just executed the code for the module, it's possible that the\n            // new exports made it ineligible for being a boundary.\n            // We only care about the case when we were _previously_ a boundary,\n            // because we already accepted this update (accidental side effect).\n            var isNoLongerABoundary = prevExports !== null;\n            if (isNoLongerABoundary) {\n                module.hot.invalidate();\n            }\n        }\n    }\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3JjL3BhZ2VzL2luZGV4LnRzeD80MWUwIl0sIm5hbWVzIjpbIkluZGV4IiwicHJvY2VzcyIsImVudiIsIkJBU0VfUEFUSCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUEsSUFBTUEsS0FBSyxHQUFHLFNBQVJBLEtBQVE7QUFBQSxzQkFDWiw4REFBQywyREFBRDtBQUFBLDRCQUdFLDhEQUFDLGlEQUFEO0FBQU0sV0FBSyxFQUFDO0FBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQUhGLGVBTUUsOERBQUMseURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQU5GLGVBU0UsOERBQUMsMkRBQUQ7QUFBVyxPQUFDLEVBQUMsTUFBYjtBQUFvQixPQUFDLEVBQUMsUUFBdEI7QUFBK0IsVUFBSSxFQUFDLE9BQXBDO0FBQTRDLFVBQUksRUFBQyxPQUFqRDtBQUF5RCxRQUFFLEVBQUMsTUFBNUQ7QUFBQSw2QkFDRTtBQUNFLGFBQUssRUFBQyxNQURSO0FBQ2UsY0FBTSxFQUFDLE1BRHRCO0FBRUUsV0FBRyxFQUFDLDJDQUZOO0FBR0UsYUFBSyxFQUFDLE9BSFI7QUFJRSxhQUFLLEVBQUMsMEZBSlI7QUFJbUcsdUJBQWU7QUFKbEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQURGO0FBQUE7QUFBQTtBQUFBO0FBQUEsYUFURixlQWtCRSw4REFBQywyREFBRDtBQUFXLE9BQUMsRUFBQyxNQUFiO0FBQW9CLFVBQUksRUFBQyxPQUF6QjtBQUFpQyxnQkFBVSxFQUFDLE1BQTVDO0FBQW1ELFFBQUUsRUFBQyxNQUF0RDtBQUE2RCxRQUFFLEVBQUMsTUFBaEU7QUFBQSw4QkFHRSw4REFBQyxxREFBRDtBQUFTLGdCQUFRLEVBQUMsS0FBbEI7QUFBd0IsVUFBRSxFQUFDLE1BQTNCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGVBSEYsZUFJRSw4REFBQyxrREFBRDtB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O0FBQUE7QUFERjtBQUFBO0FBQUE7QUFBQTtBQUFBLGlCQVJGO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxlQXBDRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsYUFsQkYsZUE4RUUsOERBQUMscUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQTlFRixlQStFRSw4REFBQyxxREFBRDtBQUFBLDZCQUNFLDhEQUFDLGtEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFERjtBQUFBO0FBQUE7QUFBQTtBQUFBLGFBL0VGO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQURZO0FBQUEsQ0FBZDs7S0FBTUgsSztBQXNGTiwrREFBZUEsS0FBZiIsImZpbGUiOiIuL3NyYy9wYWdlcy9pbmRleC50c3guanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBMaW5rIGFzIENoYWtyYUxpbmssIEltYWdlIGFzIENoYWtyYUltYWdlLCBUZXh0LCBDb2RlLCBMaXN0SXRlbSwgSGVhZGluZywgVW5vcmRlcmVkTGlzdCwgQm94LCB9IGZyb20gJ0BjaGFrcmEtdWkvcmVhY3QnXG5pbXBvcnQgeyBIZXJvIH0gZnJvbSAnY29tcG9uZW50cy9IZXJvJ1xuaW1wb3J0IHsgQ29udGFpbmVyIH0gZnJvbSAnY29tcG9uZW50cy9Db250YWluZXInXG5pbXBvcnQgTmV4dExpbmsgZnJvbSAnbmV4dC9saW5rJ1xuaW1wb3J0IHsgRGFya01vZGVTd2l0Y2ggfSBmcm9tICdjb21wb25lbnRzL0RhcmtNb2RlU3dpdGNoJ1xuaW1wb3J0IHsgTGlua3NSb3cgfSBmcm9tICdjb21wb25lbnRzL0xpbmtzUm93J1xuaW1wb3J0IHsgRm9vdGVyIH0gZnJvbSAnY29tcG9uZW50cy9Gb290ZXInXG5cbmNvbnN0IEluZGV4ID0gKCkgPT4gKFxuICA8Q29udGFpbmVyPlxuXG4gICAgey8qIEVkaXQgYXV0aG9yIGluZm8gKi99XG4gICAgPEhlcm8gdGl0bGU9XCJQaXhNYXRjaDogVW5zdXBlcnZpc2VkIERvbWFpbiBBZGFwdGF0aW9uIHZpYSBQaXhlbHdpc2UgQ29uc2lzdGVuY3kgVHJhaW5pbmdcIiAvPlxuXG4gICAgey8qIFRPRE86IEFkZCBwYXBlci9naXRodWIgbGlua3MgaGVyZSAqL31cbiAgICA8TGlua3NSb3cgLz5cblxuICAgIHsvKiBUT0RPOiBBZGQgdmlkZW8gKi99XG4gICAgPENvbnRhaW5lciB3PVwiOTB2d1wiIGg9XCI1MC42dndcIiBtYXhXPVwiNzAwcHhcIiBtYXhIPVwiMzkzcHhcIiBtYj1cIjNyZW1cIj5cbiAgICAgIDxpZnJhbWVcbiAgICAgICAgd2lkdGg9XCIxMDAlXCIgaGVpZ2h0PVwiMTAwJVwiXG4gICAgICAgIHNyYz1cImh0dHBzOi8vd3d3LnlvdXR1YmUuY29tL2VtYmVkL1NjTXpJdnhCU2k0XCJcbiAgICAgICAgdGl0bGU9XCJWaWRlb1wiXG4gICAgICAgIGFsbG93PVwiYWNjZWxlcm9tZXRlcjsgYXV0b3BsYXk7IGNsaXBib2FyZC13cml0ZTsgZW5jcnlwdGVkLW1lZGlhOyBneXJvc2NvcGU7IHBpY3R1cmUtaW4tcGljdHVyZVwiIGFsbG93RnVsbFNjcmVlbj5cbiAgICAgIDwvaWZyYW1lPlxuICAgIDwvQ29udGFpbmVyPlxuXG4gICAgPENvbnRhaW5lciB3PVwiMTAwJVwiIG1heFc9XCI0NHJlbVwiIGFsaWduSXRlbXM9XCJsZWZ0XCIgcGw9XCIxcmVtXCIgcHI9XCIxcmVtXCI+XG5cbiAgICAgIHsvKiBBYnN0cmFjdCAqL31cbiAgICAgIDxIZWFkaW5nIGZvbnRTaXplPVwiMnhsXCIgcGI9XCIxcmVtXCI+QWJzdHJhY3Q8L0hlYWRpbmc+XG4gICAgICA8VGV4dCBwYj1cIjJyZW1cIj5cbiAgICAgICAgVW5zdXBlcnZpc2VkIGRvbWFpbiBhZGFwdGF0aW9uIGlzIGEgcHJvbWlzaW5nIHRlY2huaXF1ZSBmb3Igc2VtYW50aWMgc2VnbWVudGF0aW9uIGFuZCBvdGhlciBjb21wdXRlciB2aXNpb24gdGFza3MgZm9yIHdoaWNoIGxhcmdlLXNjYWxlIGRhdGEgYW5ub3RhdGlvbiBpcyBjb3N0bHkgYW5kIHRpbWUtY29uc3VtaW5nLiBJbiBzZW1hbnRpYyBzZWdtZW50YXRpb24sIGl0IGlzIGF0dHJhY3RpdmUgdG8gdHJhaW4gbW9kZWxzIG9uIGFubm90YXRlZCBpbWFnZXMgZnJvbSBhIHNpbXVsYXRlZCAoc291cmNlKSBkb21haW4gYW5kIGRlcGxveSB0aGVtIG9uIHJlYWwgKHRhcmdldCkgZG9tYWlucy4gSW4gdGhpcyB3b3JrLCB3ZSBwcmVzZW50IGEgbm92ZWwgZnJhbWV3b3JrIGZvciB1bnN1cGVydmlzZWQgZG9tYWluIGFkYXB0YXRpb24gYmFzZWQgb24gdGhlIG5vdGlvbiBvZiB0YXJnZXQtZG9tYWluIGNvbnNpc3RlbmN5IHRyYWluaW5nLiBJbnR1aXRpdmVseSwgb3VyIHdvcmsgaXMgYmFzZWQgb24gdGhlIGlkZWEgdGhhdCBpbiBvcmRlciB0byBwZXJmb3JtIHdlbGwgb24gdGhlIHRhcmdldCBkb21haW4sIGEgbW9kZWzigJlzIG91dHB1dCBzaG91bGQgYmUgY29uc2lzdGVudCB3aXRoIHJlc3BlY3QgdG8gc21hbGwgcGVydHVyYmF0aW9ucyBvZiBpbnB1dHMgaW4gdGhlIHRhcmdldCBkb21haW4uIFNwZWNpZmljYWxseSwgd2UgaW50cm9kdWNlIGEgbmV3IGxvc3MgdGVybSB0byBlbmZvcmNlIHBpeGVsd2lzZSBjb25zaXN0ZW5jeSBiZXR3ZWVuIHRoZSBtb2RlbCdzIHByZWRpY3Rpb25zIG9uIGEgdGFyZ2V0IGltYWdlIGFuZCBhIHBlcnR1cmJlZCB2ZXJzaW9uIG9mIHRoZSBzYW1lIGltYWdlLiBJbiBjb21wYXJpc29uIHRvIHBvcHVsYXIgYWR2ZXJzYXJpYWwgYWRhcHRhdGlvbiBtZXRob2RzLCBvdXIgYXBwcm9hY2ggaXMgc2ltcGxlciwgZWFzaWVyIHRvIGltcGxlbWVudCwgYW5kIG1vcmUgbWVtb3J5LWVmZmljaWVudCBkdXJpbmcgdHJhaW5pbmcuIEV4cGVyaW1lbnRzIGFuZCBleHRlbnNpdmUgYWJsYXRpb24gc3R1ZGllcyBkZW1vbnN0cmF0ZSB0aGF0IG91ciBzaW1wbGUgYXBwcm9hY2ggYWNoaWV2ZXMgcmVtYXJrYWJseSBzdHJvbmcgcmVzdWx0cyBvbiB0d28gY2hhbGxlbmdpbmcgc3ludGhldGljLXRvLXJlYWwgYmVuY2htYXJrcywgR1RBNS10by1DaXR5c2NhcGVzIGFuZCBTWU5USElBLXRvLUNpdHlzY2FwZXMuXG4gICAgICA8L1RleHQ+XG5cbiAgICAgIHsvKiBFeGFtcGxlICovfVxuICAgICAgPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwYj1cIjFyZW1cIj5BcHByb2FjaDwvSGVhZGluZz5cbiAgICAgIDxDaGFrcmFJbWFnZSBzcmM9e2Ake3Byb2Nlc3MuZW52LkJBU0VfUEFUSCB8fCBcIlwifS9pbWFnZXMvZGlhZ3JhbS5qcGdgfSAvPlxuICAgICAgPFRleHQgYWxpZ249XCJjZW50ZXJcIiBwdD1cIjAuNXJlbVwiIHBiPVwiMC41cmVtXCIgZm9udFNpemU9XCJzbWFsbFwiPk91ciBwcm9wb3NlZCBwaXhlbHdpc2UgY29uc2lzdGVuY3kgdHJhaW5pbmcgYXBwcm9hY2guPC9UZXh0PlxuXG4gICAgICB7LyogQW5vdGhlciBTZWN0aW9uICovfVxuICAgICAgPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwdD1cIjJyZW1cIiBwYj1cIjFyZW1cIj5FeGFtcGxlczwvSGVhZGluZz5cbiAgICAgIDxDaGFrcmFJbWFnZSBzcmM9e2Ake3Byb2Nlc3MuZW52LkJBU0VfUEFUSCB8fCBcIlwifS9pbWFnZXMvZXhhbXBsZS1zeW50aGlhLmpwZ2B9IC8+XG4gICAgICA8VGV4dCBhbGlnbj1cImNlbnRlclwiIHB0PVwiMC41cmVtXCIgcGI9XCIwLjVyZW1cIiBmb250U2l6ZT1cInNtYWxsXCI+UXVhbGl0YXRpdmUgcmVzdWx0cyBvbiBTWU5USElBLXRvLUNpdHlzY2FwZXM8L1RleHQ+XG5cblxuICAgICAgey8qIENpdGF0aW9uICovfVxuICAgICAgPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwdD1cIjJyZW1cIiBwYj1cIjFyZW1cIj5DaXRhdGlvbjwvSGVhZGluZz5cbiAgICAgIDxCb3ggdz1cIjEwMCVcIiBvdmVyZmxvdz1cInNjcm9sbFwiPlxuICAgICAgICA8Q29kZSBwPVwiMC41cmVtXCIgYm9yZGVyUmFkaXVzPVwiNXB4XCIgPlxuICAgICAgICAgIHsvKiB3PVwiMTUwJVwiPiAqL31cbiAgICAgICAgICBAaW5wcm9jZWVkaW5ncyYjMTIzOyA8YnIgLz5cbiAgICAgICAgICAmbmJzcDsmbmJzcDsmbmJzcDsmbmJzcDt5dTIwMjFwbGVub2N0cmVlcywgPGJyIC8+XG4gICAgICAgICAgJm5ic3A7Jm5ic3A7Jm5ic3A7Jm5ic3A7dGl0bGU9JiMxMjM7UGl4TWF0Y2g6IFVuc3VwZXJ2aXNlZCBEb21haW4gQWRhcHRhdGlvbiB2aWEgUGl4ZWx3aXNlIENvbnNpc3RlbmN5IFRyYWluaW5nJiMxMjU7IDxiciAvPlxuICAgICAgICAgICZuYnNwOyZuYnNwOyZuYnNwOyZuYnNwO2F1dGhvcj0mIzEyMztMdWtlIE1lbGFzLUt5cmlhemkgYW5kIEFyanVuIEsuIE1hbnJhaSYjMTI1OyA8YnIgLz5cbiAgICAgICAgICAmbmJzcDsmbmJzcDsmbmJzcDsmbmJzcDt5ZWFyPSYjMTIzOzIwMjEmIzEyNTsgPGJyIC8+XG4gICAgICAgICAgJm5ic3A7Jm5ic3A7Jm5ic3A7Jm5ic3A7Ym9va3RpdGxlPSYjMTIzO0NWUFImIzEyNTsgPGJyIC8+XG4gICAgICAmIzEyNTtcbiAgICAgIDwvQ29kZT5cbiAgICAgIDwvQm94PlxuXG4gICAgICB7LyogUmVsYXRlZCBXb3JrICovfVxuICAgICAgPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwdD1cIjJyZW1cIiBwYj1cIjFyZW1cIj5SZWxhdGVkIFdvcms8L0hlYWRpbmc+XG4gICAgICA8VW5vcmRlcmVkTGlzdD5cbiAgICAgICAgPExpc3RJdGVtPlxuICAgICAgICAgIDxUZXh0IGNvbG9yPVwiYmx1ZVwiPlxuICAgICAgICAgICAgPE5leHRMaW5rIGhyZWY9XCIjXCIgcGFzc0hyZWY9e3RydWV9PlxuICAgICAgICAgICAgICBGaXJzdCBwYXBlclxuICAgICAgICAgICAgPC9OZXh0TGluaz5cbiAgICAgICAgICA8L1RleHQ+XG4gICAgICAgIDwvTGlzdEl0ZW0+XG4gICAgICAgIDxMaXN0SXRlbT5cbiAgICAgICAgICA8VGV4dCBjb2xvcj1cImJsdWVcIj5cbiAgICAgICAgICAgIDxOZXh0TGluayBocmVmPVwiI1wiIHBhc3NIcmVmPXt0cnVlfT5cbiAgICAgICAgICAgICAgU2Vjb25kIHBhcGVyXG4gICAgICAgICAgICA8L05leHRMaW5rPlxuICAgICAgICAgIDwvVGV4dD5cbiAgICAgICAgPC9MaXN0SXRlbT5cbiAgICAgIDwvVW5vcmRlcmVkTGlzdD5cblxuICAgICAgey8qIEFja25vd2xlZGdlbWVudHMgKi99XG4gICAgICB7LyogPEhlYWRpbmcgZm9udFNpemU9XCIyeGxcIiBwdD1cIjJyZW1cIiBwYj1cIjFyZW1cIj5BY2tub3dsZWRnZW1lbnRzPC9IZWFkaW5nPlxuICAgICAgPFRleHQgPlxuICAgICAgICBXZSB0aGFuayB4eXogZm9yIGFiYy4uLlxuICAgICAgPC9UZXh0PiAqL31cbiAgICA8L0NvbnRhaW5lcj5cblxuICAgIDxEYXJrTW9kZVN3aXRjaCAvPlxuICAgIDxGb290ZXI+XG4gICAgICA8VGV4dD48L1RleHQ+XG4gICAgPC9Gb290ZXI+XG4gIDwvQ29udGFpbmVyID5cbilcblxuZXhwb3J0IGRlZmF1bHQgSW5kZXhcbiJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///./src/pages/index.tsx\n");

/***/ })

});