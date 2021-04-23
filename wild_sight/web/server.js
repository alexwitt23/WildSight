const express = require('express');
const path = require('path');
app = express();
app.use(express.static(path.join(__dirname, 'dist')));
app.use('public', express.static(path.join(__dirname, 'public')));
const port = process.env.PORT || 5000;
console.log("Express server listening on port %d", port)


var nStatic = require('node-static');
var fileServer = new nStatic.Server(
    {headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type'
      }}
);

require('http').createServer(function (request, response) {
    request.addListener('end', function () {
        fileServer.serve(request, response);
    }).resume();
}).listen(8080);

app.listen(port);