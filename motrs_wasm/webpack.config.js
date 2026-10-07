const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");

module.exports = {
    entry: {
        app: "./index.js",
    },
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "[name].js",
    },
    plugins: [
        new CleanWebpackPlugin(),
        new HtmlWebpackPlugin({ template: "./html/index.html" }),
    ],
    module: {
        rules: [{ test: /\.wasm$/, type: "asset/resource" }],
    },
    devServer: {
        host: "127.0.0.1",
        port: 8080,
    },
    mode: "development",
};
