/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("pages/index",{

/***/ "./src/components/Hero.tsx":
/*!*********************************!*\
  !*** ./src/components/Hero.tsx ***!
  \*********************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"Hero\": function() { return /* binding */ Hero; }\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"./node_modules/react/jsx-dev-runtime.js\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var _chakra_ui_react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @chakra-ui/react */ \"./node_modules/@chakra-ui/react/dist/esm/index.js\");\n/* harmony import */ var next_app__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/app */ \"./node_modules/next/app.js\");\n/* harmony import */ var next_app__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_app__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/link */ \"./node_modules/next/link.js\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_2__);\n/* module decorator */ module = __webpack_require__.hmd(module);\n\n\nvar _jsxFileName = \"/Users/user/Documents/projects/experiments/pixmatch/src/components/Hero.tsx\",\n    _this = undefined;\n\n\n\n\nvar institutions = {\n  1: \"Harvard University\",\n  2: \"Oxford University\",\n  3: \"Boston Children's Hospital\"\n};\nvar authors = [{\n  'name': 'Luke Melas-Kyraizi',\n  'institutions': [1, 2],\n  'url': \"https://lukemelas.github.io/\"\n}, {\n  'name': 'Arjun K. Manrai',\n  'institutions': [1, 3],\n  'url': \"https://www.childrenshospital.org/research/researchers/m/arjun-manrai\"\n}];\nvar Hero = function Hero(_ref) {\n  var title = _ref.title;\n  return /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_app__WEBPACK_IMPORTED_MODULE_1__.Container, {\n    children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_3__.Heading, {\n      fontSize: \"3xl\",\n      pt: \"3rem\",\n      children: title\n    }, void 0, false, {\n      fileName: _jsxFileName,\n      lineNumber: 26,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_3__.Wrap, {\n      justify: \"center\",\n      pt: \"1rem\",\n      fontSize: \"xl\",\n      children: authors.map(function (author) {\n        return /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"span\", {\n          children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_2___default()), {\n            href: author.url,\n            passHref: true,\n            children: /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_3__.Link, {\n              children: author.name\n            }, void 0, false, {\n              fileName: _jsxFileName,\n              lineNumber: 32,\n              columnNumber: 15\n            }, _this)\n          }, void 0, false, {\n            fileName: _jsxFileName,\n            lineNumber: 31,\n            columnNumber: 13\n          }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"sup\", {\n            children: [\" \", author.institutions.toString()]\n          }, void 0, true, {\n            fileName: _jsxFileName,\n            lineNumber: 34,\n            columnNumber: 13\n          }, _this)]\n        }, author.name, true, {\n          fileName: _jsxFileName,\n          lineNumber: 30,\n          columnNumber: 11\n        }, _this);\n      })\n    }, \"authors\", false, {\n      fileName: _jsxFileName,\n      lineNumber: 27,\n      columnNumber: 5\n    }, _this), /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(_chakra_ui_react__WEBPACK_IMPORTED_MODULE_3__.Wrap, {\n      justify: \"center\",\n      pt: \"1rem\",\n      children: Object.entries(institutions).map(function (tuple) {\n        return /*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"span\", {\n          children: [/*#__PURE__*/(0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"sup\", {\n            children: [tuple[0], \"  \"]\n          }, void 0, true, {\n            fileName: _jsxFileName,\n            lineNumber: 43,\n            columnNumber: 13\n          }, _this), tuple[1]]\n        }, void 0, true, {\n          fileName: _jsxFileName,\n          lineNumber: 42,\n          columnNumber: 11\n        }, _this);\n      })\n    }, \"institutions\", false, {\n      fileName: _jsxFileName,\n      lineNumber: 39,\n      columnNumber: 5\n    }, _this)]\n  }, void 0, true, {\n    fileName: _jsxFileName,\n    lineNumber: 25,\n    columnNumber: 3\n  }, _this);\n};\n_c = Hero;\nHero.defaultProps = {\n  title: 'Academic Project Template'\n};\n\nvar _c;\n\n$RefreshReg$(_c, \"Hero\");\n\n;\n    var _a, _b;\n    // Legacy CSS implementations will `eval` browser code in a Node.js context\n    // to extract CSS. For backwards compatibility, we need to check we're in a\n    // browser context before continuing.\n    if (typeof self !== 'undefined' &&\n        // AMP / No-JS mode does not inject these helpers:\n        '$RefreshHelpers$' in self) {\n        var currentExports = module.__proto__.exports;\n        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;\n        // This cannot happen in MainTemplate because the exports mismatch between\n        // templating and execution.\n        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n        // A module can be accepted automatically based on its exports, e.g. when\n        // it is a Refresh Boundary.\n        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n            // Save the previous exports on update so we can compare the boundary\n            // signatures.\n            module.hot.dispose(function (data) {\n                data.prevExports = currentExports;\n            });\n            // Unconditionally accept an update to this module, we'll check if it's\n            // still a Refresh Boundary later.\n            module.hot.accept();\n            // This field is set when the previous version of this module was a\n            // Refresh Boundary, letting us know we need to check for invalidation or\n            // enqueue an update.\n            if (prevExports !== null) {\n                // A boundary can become ineligible if its exports are incompatible\n                // with the previous exports.\n                //\n                // For example, if you add/remove/change exports, we'll want to\n                // re-execute the importing modules, and force those components to\n                // re-render. Similarly, if you convert a class component to a\n                // function, we want to invalidate the boundary.\n                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {\n                    module.hot.invalidate();\n                }\n                else {\n                    self.$RefreshHelpers$.scheduleUpdate();\n                }\n            }\n        }\n        else {\n            // Since we just executed the code for the module, it's possible that the\n            // new exports made it ineligible for being a boundary.\n            // We only care about the case when we were _previously_ a boundary,\n            // because we already accepted this update (accidental side effect).\n            var isNoLongerABoundary = prevExports !== null;\n            if (isNoLongerABoundary) {\n                module.hot.invalidate();\n            }\n        }\n    }\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3JjL2NvbXBvbmVudHMvSGVyby50c3g/YTIyZiJdLCJuYW1lcyI6WyJpbnN0aXR1dGlvbnMiLCJhdXRob3JzIiwiSGVybyIsInRpdGxlIiwibWFwIiwiYXV0aG9yIiwidXJsIiwibmFtZSIsInRvU3RyaW5nIiwiT2JqZWN0IiwiZW50cmllcyIsInR1cGxlIiwiZGVmYXVsdFByb3BzIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUVBLElBQU1BLFlBQVksR0FBRztBQUNuQixLQUFHLG9CQURnQjtBQUVuQixLQUFHLG1CQUZnQjtBQUduQixLQUFHO0FBSGdCLENBQXJCO0FBTUEsSUFBTUMsT0FBTyxHQUFHLENBQ2Q7QUFDRSxVQUFRLG9CQURWO0FBRUUsa0JBQWdCLENBQUMsQ0FBRCxFQUFJLENBQUosQ0FGbEI7QUFHRSxTQUFPO0FBSFQsQ0FEYyxFQU1kO0FBQ0UsVUFBUSxpQkFEVjtBQUVFLGtCQUFnQixDQUFDLENBQUQsRUFBSSxDQUFKLENBRmxCO0FBR0UsU0FBTztBQUhULENBTmMsQ0FBaEI7QUFhTyxJQUFNQyxJQUFJLEdBQUcsU0FBUEEsSUFBTztBQUFBLE1BQUdDLEtBQUgsUUFBR0EsS0FBSDtBQUFBLHNCQUNsQiw4REFBQywrQ0FBRDtBQUFBLDRCQUNFLDhEQUFDLHFEQUFEO0FBQVMsY0FBUSxFQUFDLEtBQWxCO0FBQXdCLFFBQUUsRUFBQyxNQUEzQjtBQUFBLGdCQUFtQ0E7QUFBbkM7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQURGLGVBRUUsOERBQUMsa0RBQUQ7QUFBTSxhQUFPLEVBQUMsUUFBZDtBQUF1QixRQUFFLEVBQUMsTUFBMUI7QUFBaUMsY0FBUSxFQUFDLElBQTFDO0FBQUEsZ0JBRUlGLE9BQU8sQ0FBQ0csR0FBUixDQUFZLFVBQUNDLE1BQUQ7QUFBQSw0QkFDVjtBQUFBLGtDQUNFLDhEQUFDLGtEQUFEO0FBQVUsZ0JBQUksRUFBRUEsTUFBTSxDQUFDQyxHQUF2QjtBQUE0QixvQkFBUSxFQUFFLElBQXRDO0FBQUEsbUNBQ0UsOERBQUMsa0RBQUQ7QUFBQSx3QkFBYUQsTUFBTSxDQUFDRTtBQUFwQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBREY7QUFBQTtBQUFBO0FBQUE7QUFBQSxtQkFERixlQUlFO0FBQUEsNEJBQU9GLE1BQU0sQ0FBQ0wsWUFBUCxDQUFvQlEsUUFBcEIsRUFBUDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsbUJBSkY7QUFBQSxXQUFXSCxNQUFNLENBQUNFLElBQWxCO0FBQUE7QUFBQTtBQUFBO0FBQUEsaUJBRFU7QUFBQSxPQUFaO0FBRkosT0FBbUQsU0FBbkQ7QUFBQTtBQUFBO0FBQUE7QUFBQSxhQUZGLGVBY0UsOERBQUMsa0RBQUQ7QUFBTSxhQUFPLEVBQUMsUUFBZDtBQUF1QixRQUFFLEVBQUMsTUFBMUI7QUFBQSxnQkFFSUUsTUFBTSxDQUFDQyxPQUFQLENBQWVWLFlBQWYsRUFBNkJJLEdBQTdCLENBQWlDLFVBQUFPLEtBQUs7QUFBQSw0QkFDcEM7QUFBQSxrQ0FDRTtBQUFBLHVCQUFNQSxLQUFLLENBQUMsQ0FBRCxDQUFYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxtQkFERixFQUVHQSxLQUFLLENBQUMsQ0FBRCxDQUZSO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxpQkFEb0M7QUFBQSxPQUF0QztBQUZKLE9BQXFDLGNBQXJDO0FBQUE7QUFBQTtBQUFBO0FBQUEsYUFkRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FEa0I7QUFBQSxDQUFiO0tBQU1ULEk7QUE0QmJBLElBQUksQ0FBQ1UsWUFBTCxHQUFvQjtBQUNsQlQsT0FBSyxFQUFFO0FBRFcsQ0FBcEIiLCJmaWxlIjoiLi9zcmMvY29tcG9uZW50cy9IZXJvLnRzeC5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IEhlYWRpbmcsIFdyYXAsIExpbmsgYXMgQ2hha3JhTGluayB9IGZyb20gJ0BjaGFrcmEtdWkvcmVhY3QnXG5pbXBvcnQgeyBDb250YWluZXIgfSBmcm9tICduZXh0L2FwcCdcbmltcG9ydCBOZXh0TGluayBmcm9tIFwibmV4dC9saW5rXCJcblxuY29uc3QgaW5zdGl0dXRpb25zID0ge1xuICAxOiBcIkhhcnZhcmQgVW5pdmVyc2l0eVwiLFxuICAyOiBcIk94Zm9yZCBVbml2ZXJzaXR5XCIsXG4gIDM6IFwiQm9zdG9uIENoaWxkcmVuJ3MgSG9zcGl0YWxcIixcbn1cblxuY29uc3QgYXV0aG9ycyA9IFtcbiAge1xuICAgICduYW1lJzogJ0x1a2UgTWVsYXMtS3lyYWl6aScsXG4gICAgJ2luc3RpdHV0aW9ucyc6IFsxLCAyXSxcbiAgICAndXJsJzogXCJodHRwczovL2x1a2VtZWxhcy5naXRodWIuaW8vXCJcbiAgfSxcbiAge1xuICAgICduYW1lJzogJ0FyanVuIEsuIE1hbnJhaScsXG4gICAgJ2luc3RpdHV0aW9ucyc6IFsxLCAzXSxcbiAgICAndXJsJzogXCJodHRwczovL3d3dy5jaGlsZHJlbnNob3NwaXRhbC5vcmcvcmVzZWFyY2gvcmVzZWFyY2hlcnMvbS9hcmp1bi1tYW5yYWlcIlxuICB9LFxuXVxuXG5leHBvcnQgY29uc3QgSGVybyA9ICh7IHRpdGxlIH06IHsgdGl0bGU6IHN0cmluZyB9KSA9PiAoXG4gIDxDb250YWluZXI+XG4gICAgPEhlYWRpbmcgZm9udFNpemU9XCIzeGxcIiBwdD1cIjNyZW1cIj57dGl0bGV9PC9IZWFkaW5nPlxuICAgIDxXcmFwIGp1c3RpZnk9XCJjZW50ZXJcIiBwdD1cIjFyZW1cIiBmb250U2l6ZT1cInhsXCIga2V5PVwiYXV0aG9yc1wiPlxuICAgICAge1xuICAgICAgICBhdXRob3JzLm1hcCgoYXV0aG9yKSA9PlxuICAgICAgICAgIDxzcGFuIGtleT17YXV0aG9yLm5hbWV9PlxuICAgICAgICAgICAgPE5leHRMaW5rIGhyZWY9e2F1dGhvci51cmx9IHBhc3NIcmVmPXt0cnVlfT5cbiAgICAgICAgICAgICAgPENoYWtyYUxpbms+e2F1dGhvci5uYW1lfTwvQ2hha3JhTGluaz5cbiAgICAgICAgICAgIDwvTmV4dExpbms+XG4gICAgICAgICAgICA8c3VwPiB7YXV0aG9yLmluc3RpdHV0aW9ucy50b1N0cmluZygpfTwvc3VwPlxuICAgICAgICAgIDwvc3Bhbj5cbiAgICAgICAgKVxuICAgICAgfVxuICAgIDwvV3JhcD5cbiAgICA8V3JhcCBqdXN0aWZ5PVwiY2VudGVyXCIgcHQ9XCIxcmVtXCIga2V5PVwiaW5zdGl0dXRpb25zXCI+XG4gICAgICB7XG4gICAgICAgIE9iamVjdC5lbnRyaWVzKGluc3RpdHV0aW9ucykubWFwKHR1cGxlID0+XG4gICAgICAgICAgPHNwYW4+XG4gICAgICAgICAgICA8c3VwPnt0dXBsZVswXX0gIDwvc3VwPlxuICAgICAgICAgICAge3R1cGxlWzFdfVxuICAgICAgICAgIDwvc3Bhbj5cbiAgICAgICAgKVxuICAgICAgfVxuICAgIDwvV3JhcD5cbiAgPC9Db250YWluZXI+XG4pXG5cbkhlcm8uZGVmYXVsdFByb3BzID0ge1xuICB0aXRsZTogJ0FjYWRlbWljIFByb2plY3QgVGVtcGxhdGUnLFxufSJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///./src/components/Hero.tsx\n");

/***/ })

});