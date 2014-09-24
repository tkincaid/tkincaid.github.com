/******************************************************************************
 * DataRobot Testing Framework: test.js
 *
 *  Author: David Lapointe
 *
 * Copyright 2013, DataRobot Inc
 ******************************************************************************/
"use strict";

var jsdom = require("jsdom");
var fs = require("fs");
var path = require("path");
var jasmine = require("jasmine-node");
jasmine.CATCH_EXCEPTIONS = false;
jasmine.VERBOSE = true;
var jasmineEnv = jasmine.getEnv();
jasmineEnv.addReporter(new jasmine.JUnitXmlReporter("reports/"));
jasmineEnv.addReporter(new jasmine.ConsoleReporter());
    /*{
    "color":1,
    "onComplete":function(){
        if(jasmineEnv.currentRunner().results().failedCount < 1)
            process.exit(0);
        else
            process.exit(1);
    }
}));*/

var html = fs.readFileSync("../../MMApp/templates/main.html", "utf8");
html = html.replace("baseUrl: \"static/scripts/\",", "baseUrl: \"file://"+path.resolve("../../MMApp/static/scripts/")+"\",");
var window, errors;
var documentReady = false;
var config = {
    "html":html,
    "url":"file://"+path.resolve("../../MMApp/index"),
    "features":{
        "FetchExternalResources":["script"],
        "ProcessExternalResources":["script"],
    },
    "done":function(_errors, _window){
        errors = _errors;
        window = _window;
        window.onerror = function(){
            var args = Array.prototype.slice.call(arguments);
            args.unshift("STDERR: ");
            args.unshift((new Date()).getTime()/1000);
            console.log.apply(window, args);
            //process.exit();
        };
        window.console.log = function(){
            var args = Array.prototype.slice.call(arguments);
            args.unshift("STDOUT: ");
            args.unshift((new Date()).getTime()/1000);
            console.log.apply(window, args);
        };
        window.document.documentElement.clientHeight = 400;
        window.document.documentElement.clientWidth = 600;
        window.document.ready = function(){
            spoof_ajax();
            window.require(["main"], function() {
                documentReady = true;
                jasmineEnv.execute();
            });
        }
    }
};
jsdom.env(config);

var ajax_url_log = [];
var tmp_queue = null;
var tmp_models = [];
var service_call = null;
function spoof_ajax($){
    var loggedusername = "";
    var spoofed_ajax_func = function(options){
        ajax_url_log.unshift(options.url);
        console.log("AJAX: "+options.url);
        switch(options.url){
            case '/account/username':
                // TODO: This shouldn't need to be delayed
                setTimeout(function(){
                    if(loggedusername){
                        try{
                            options.success({"username": loggedusername, "sc":0});
                        }catch(e){
                            console.log(e);
                        }
                    }else
                        options.success({"username": "", "sc":0});
                }, 100);
                break;
            case '/account/login':
                setTimeout(function(){
                    loggedusername = JSON.parse(options.data)["username"];
                    options.success({"error": 0, "message":"Log In Successful"});
                }, 100);
                break;
            case '/projects':
                var json = [{"project_name": "Untitled Project", "created": "2013-07-03 18:48:40.219000", "filename": "kickcars.rawdata.csv", "mode": 1, "active": 1, "_id": "51d47208700dc51833e0a30c", "stage": "aim"}];
                setTimeout(function(){
                    options.success(json);
                }, 100);
                break;
            case '/dataset/51d47208700dc51833e0a30c':
                var json = JSON.parse(fs.readFileSync("testfiles/eda.json", "utf8"));
                setTimeout(function(){
                    options.success(json);
                    options.complete(json);
                }, 100);
                break;
            case '/ds/51d47208700dc51833e0a30c/models':
                setTimeout(function(){
                    options.success(tmp_models);
                }, 150);
                break;
            case '/aim':
                setTimeout(function(){
                    options.success({"status": ""});
                }, 50);
                break;
            case '/aim2/51d47208700dc51833e0a30c':
                setTimeout(function(){
                    options.success({"status": "ready"});
                }, 50);
                break;
            case '/eda2/51d47208700dc51833e0a30c':
                // TODO: Large sample with multiple calls
                var json = JSON.parse(fs.readFileSync("testfiles/eda2.json", "utf8"));
                setTimeout(function(){
                    options.success(json);
                    options.complete(json);
                }, 150);
                break;
            case '/queue':
                if(!tmp_queue)
                    tmp_queue = JSON.parse(fs.readFileSync("testfiles/queue.json", "utf8"));
                setTimeout(function(){
                    options.success(tmp_queue);
                }, 100);
                break;
            case '/service':
                var json;
                /*switch(service_calls){
                    case 0:
                        json = {"qid": 1, "change": "started"};
                        tmp_queue[1].status = "inprogress";
                        break;
                    case 5:
                        json = {"qid": 1, "change": "completed"};
                        tmp_queue[1].status = "inprogress";
                        break;
                    default:
                        json = {"change": "none"};
                        break;
                }
                ++service_calls;*/
                if(!service_call){
                    json = {"change": "none"};
                }else{
                    json = service_call;
                    service_call = null;
                }
                setTimeout(function(){
                    options.success(json);
                }, 100);
                break;
            case '/resource_usage':
                var json = {"mem":{}, "cpu":{}};
                var totalmem = 100;
                for(var q in tmp_queue){
                    if(tmp_queue[q]["status"] == "inprogress"){
                        var cmem = Math.random()*totalmem;
                        totalmem -= cmem;
                        json["mem"][tmp_queue[q]["_id"]] = [12289437696, cmem];
                        json["cpu"][tmp_queue[q]["_id"]] = [Math.random()*100,
                            Math.random()*100, Math.random()*100, Math.random()*100];
                    }
                }
                setTimeout(function(){
                    options.success(json);
                    options.complete();
                }, 300);
                break;
            default:
                if(options.url.lastIndexOf("/queue/", 0) === 0){
                    // /queue/<id>
                    var qid = parseInt(options.url.substring(7), 10);
                    for(var q in tmp_queue){
                        if(tmp_queue[q]["_id"] != qid)
                            continue;
                        tmp_queue.splice(q, 1);
                        break;
                    }
                    var json = {'message':'OK'};
                    setTimeout(function(){
                        options.success(json);
                    }, 100);
                    break;
                }else if(options.url.lastIndexOf("/ds/51d47208700dc51833e0a30c/models/", 0) === 0){
                    var json = {'message':'OK'};
                    setTimeout(function(){
                        options.success(json);
                    }, 100);
                    break;
                }
                console.log("Did not handle AJAX: ");
                console.log(options);
                break;
        }
    };
    spyOn(window.$, "ajax", true).andCallFake(spoofed_ajax_func);
}

/* Tests */

describe("tests", function(){
    it("should wait for the window to be generated", function(){
        runs(function(){ });
        waitsFor(function(){
            return documentReady;
        }, 5000, "window being generated");
        runs(function(){ });
    });
    it("should not have generated errors", function(){
        expect(errors).toBeNull();
    });
    it("should have a document", function(){
        expect(window.document).toBeDefined();
    });
});

var $;
describe("jQuery", function() {
    it("should have the $ function", function(){
        $ = window.$;
        $.fx.off = true
        $.fn.reverse = [].reverse;
        expect($).toBeDefined();
        expect($).toBe(window.jQuery);
    });
});

describe("require", function() {
    it("should have the require function", function(){
        expect(window.require).toBeDefined();
        expect(window.require).toBe(window.requirejs);
    });
});

var app;
describe("the app view", function(){
    it("should exist", function(){
        app = window.app;
        expect(app).toBeDefined();
    });
    it("should resize", function(){
        /* This lamely only checks the body height. TODO: Add more CSS checks */
        runs(function(){
            window.document.documentElement.clientHeight = 800;
            window.document.documentElement.clientWidth = 1000;
            $(window).trigger("resize");
        });
        waitsFor(function(){
            return $('body').height()==800;
        }, "body should be 800px", 500);
        runs(function(){
            window.document.documentElement.clientHeight = 400;
            window.document.documentElement.clientWidth = 600;
            $(window).trigger("resize");
        });
        waitsFor(function(){
            return $('body').height()==400;
        }, "body should be 400px", 500);
        runs(function(){
            expect($('body').height()).toEqual(400);
        });
    });
});

describe("usage", function(){
    it("should not be running", function(){
        var usage = window.requirejs("usage");
        expect(usage.running).toBeFalsy();
    });
});

var nav_view;
describe("the nav view", function(){
    it("should exist", function(){
        nav_view = app.nav_view;
        expect(nav_view).toBeDefined();
    });
    it("should not be visible", function(){
        // Slowing everything down for now, so this needs to wait
        waitsFor(function(){
            return !$('#Nav_Bar').is(":visible");
        }, 200);
        runs(function(){
            expect($('#Nav_Bar').is(":visible")).toBeFalsy();
        });
    });
    it("should go to the front page when clicked", function(){
        expect($('.logo').is(":visible")).toBeTruthy();
        $(".logo").trigger("click");
        expect(nav_view.current_state).toBe(0);
    });
    /* FIXME: Test doesn't work, oddly.
        .is(":visible") handles differently than in browser */
    xit("has non visible components", function(){
        expect($('#eda_link').is(":visible")).toBeFalsy();
        expect($('#leaderboard_link').is(":visible")).toBeFalsy();
    });
});

var user_view;
describe("the user view", function(){
    it("should exist", function(){
        user_view = app.user_view;
        expect(user_view).toBeDefined();
    });
    it("should bring up a login box when clicked", function(){
        runs(function(){
            $("#user_login_link").trigger("click");
        });
        waitsFor(function(){
            return ($("#login_form").length == 1);
        }, 250);
        runs(function(){
            expect($("#login_form").length).toBe(1);
        });
    });
    describe("login", function(){
        it("should err if username is too short", function(){
            $('#user_info #username').val("Ja");
            $("#login_btn").trigger("click");
            expect($('#user_info #username_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        it("should err if username is too long", function(){
            $('#user_info #username').val((new Array( 50 )).join( "A" ));
            $("#login_btn").trigger("click");
            expect($('#user_info #username_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        it("should not show an error if a valid user is used after an error", function(){
            $('#user_info #username').val("Jasmine");
            $("#login_btn").trigger("click");
            expect($('#user_info #username_validation').html()).toBeFalsy();
        });
        xit("should err if username contains dangerous strings", function(){
            $('#user_info #username').val(String.fromCharCode(27)+'[31mJasmine');
            $("#login_btn").trigger("click");
            expect($('#user_info #username_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
            
            $('#user_info #username').val("<script>Jasmine</script>");
            $("#login_btn").trigger("click");
            expect($('#user_info #username_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        it("should err if password is too short", function(){
            $('#user_info #password').val("Ja");
            $("#login_btn").trigger("click");
            expect($('#user_info #password_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        it("should clear password after submission", function(){
            expect($('#user_info #password').val()).toBeFalsy();
        });
        it("should err if password is too long", function(){
            $('#user_info #password').val((new Array( 500 )).join( "A" ));
            $("#login_btn").trigger("click");
            expect($('#user_info #password_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        xit("should err if password contains dangerous strings", function(){
            $('#user_info #password').val(String.fromCharCode(27)+'[31mJasmine');
            $("#login_btn").trigger("click");
            expect($('#user_info #password_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
            
            $('#user_info #password').val("<script>Jasmine</script>");
            $("#login_btn").trigger("click");
            expect($('#user_info #password_validation').html()).toBeTruthy();
            expect(ajax_url_log).not.toContain("/account/login");
        });
        // FIXME: Some AJAX nonsense occurs in here.
        it("should call login if username/password is valid", function(){
            $('#user_info #username').val("Jasmine");
            $('#user_info #password').val("JasPassword");
            runs(function(){
                $("#login_btn").trigger("click");
            });
            waitsFor(function(){
                return ($.inArray("/account/login", ajax_url_log) > -1);
            }, "login ajax should be triggered", 500);
            runs(function(){
                expect($('#user_info #username_validation').html()).toBeFalsy();
                expect($('#user_info #password_validation').html()).toBeFalsy();
            });
        });
        it("should clear password after submission", function(){
            expect($('#user_info #password').val()).toBeFalsy();
        });
        it("should be signed in", function(){
            waitsFor(function(){
                return ($("#user_name").length == 1);
            }, 500);
            runs(function(){
                expect($("#user_name").length).toBe(1);
            });
            expect(ajax_url_log).toContain("/account/username");
        });
    });
});

var eda_view;
describe("EDA View", function(){
    it("should exist", function(){
        waitsFor(function(){
            return nav_view.eda_view;
        }, "eda_view should be defined", 1000);
        runs(function(){
            eda_view = nav_view.eda_view;
            expect(eda_view).toBeDefined();
        });
    });
    it("should be visible when aim is closed", function(){
        expect($("#main").hasClass("transparent")).toBeTruthy();
        expect($("#aim_close").length).not.toBe(0);
        $("#aim_close").trigger("click");
        expect($("#main").hasClass("transparent")).toBeFalsy();
        expect($("#aim_popup").find("div").length).toBe(0);
    });
    var targets = {
        "#EDA_menu>li":[
            "eda_menu", // These need to be called every element
            ["show_stats", "renderProfile"] // These need to be called at least once for every set of elements
        ],
        ".col_name":[
            "highlight_on",
            "highlight_off",
            "showProfile",
        ],
    };
    for(var target in targets){
        it("should have "+target, function(){
            waitsFor(function(){
                return $(target).length;
            }, "should exist in DOM", 1000);
            runs(function(){
                expect($(target).length).not.toBe(0);
            });
        });
        it("should be able to safely click "+target, function(){
            var opts = [];
            var reqs = [];
            var targetlen = 0;
            runs(function(){
                for(var f in targets[target]){
                    var func = targets[target][f];
                    if($.isArray(func)){
                        for(var g in func){
                            opts.unshift(spyOn(eda_view, func[g]).andCallThrough());
                            
                        }
                    }else{
                        reqs.unshift(spyOn(eda_view, func).andCallThrough());
                    }
                }
                eda_view.delegateEvents(); // reset functions in View to call the spies
                targetlen = $(target).length;
                $(target).reverse().each(function(i){
                    $(this).trigger("mouseenter");
                    waits(20);
                    $(this).trigger("click");
                    waits(100);
                    $(this).trigger("mouseleave");
                    waits(30);
                });
            });
            runs(function(){
                for(var f in opts){
                    expect(opts[f]).toHaveBeenCalled();
                }
                for(f in reqs){
                    expect(reqs[f].callCount).toBe(targetlen);
                }
            });
        });
    }
});

describe("AIM Selection", function(){
    it("should occur before the queue is running", function(){
        expect(nav_view.queue_running).toBeFalsy();
    });
    it("should popup if still available", function(){
        runs(function(){ });
        waitsFor(function(){
            return $("#aim_view_select").length;
        }, 1000, "aim should popup");
        runs(function(){
            expect($("#aim_view_select").length).toBe(1);
            $("#aim_view_select").trigger("click"); 
        });
    });
    it("should err if invalid input is given", function(){
        $('#targetin').val("z");
        $('.mode_selector').each(function(){
            $(this).trigger("click");
            waits(30);
            expect($('#targeterror').html()).toBeTruthy();
        });
    });
    it("should process when a valid data is input", function(){
        runs(function(){
            spyOn(nav_view, "request_univariates").andCallThrough();
            $('#targetin').val("y");
            // click on Semi-Automatically
            $($('.mode_selector').get(1)).trigger("click");
        });
        waitsFor(function(){
            return nav_view.request_univariates.wasCalled;
        }, 1000, "Univariates requested");
        runs(function(){
            expect(nav_view.request_univariates.wasCalled).toBeTruthy();
        });
    });
});

function processQueueItem(qid){
    service_call = {"qid": qid, "change": "started"};
    for(var q in tmp_queue){
        if(tmp_queue[q]["_id"] != qid)
            continue;
        tmp_queue[q].status = "inprogress";
        break;
    }
}

function completeQueueItem(qid){
    service_call = {"qid": qid, "change": "completed"};
    for(var q in tmp_queue){
        if(tmp_queue[q]["_id"] != qid)
            continue;
        tmp_queue.splice(q, 1);
        break;
    }
    tmp_models = JSON.parse(fs.readFileSync("testfiles/models.json", "utf8"));
}

function errQueueItem(qid){
    service_call = {"qid": qid, "change": "completed"};
    for(var q in tmp_queue){
        if(tmp_queue[q]["_id"] != qid)
            continue;
        tmp_queue[q].status = "error";
        tmp_queue[q].error_log = [
            "Traceback (most recent call last):\n", "<Error> thrown\n", "", "\n"
        ];
        break;
    }
}

var queue_view;
describe("Queue", function(){
    it("should exist", function(){
         queue_view = nav_view.queue_view;
         expect(queue_view).toBeDefined();
    });
    it("should be running", function(){
        runs(function(){ });
        waitsFor(function(){
            return nav_view.queue_running;
        }, "Queue running", 1000);
        runs(function(){
            expect(nav_view.queue_running).toBeTruthy();
        });
    });
    it("should call service", function(){
        spyOn(nav_view, "queue_serve").andCallThrough();
        runs(function(){ });
        /*waitsFor(function(){
            return nav_view.show_queue.wasCalled;
        }, 1000, "Calling show_queue");*/
        waitsFor(function(){
            if($.inArray("/service", ajax_url_log) == -1)
                return false;
            return nav_view.queue_serve.wasCalled;
        }, 1000, "Calling service");
        runs(function(){
            expect(nav_view.queue_serve.wasCalled).toBeTruthy();
            expect(ajax_url_log).toContain("/service");
        });
    });
    it("should trigger the addOne queue event", function(){
        runs(function(){
            spyOn(queue_view, "addOne").andCallThrough();
            spyOn(queue_view, "addAll").andCallThrough();
            queue_view.delegateEvents();
            processQueueItem(1);
        });
        waitsFor(function(){
            return queue_view.addOne.wasCalled;
        }, 1200, "calling Queue addOne");
        runs(function(){
            queue_view.addOne.wasCalled = false;
            processQueueItem(2);
        });
        waitsFor(function(){
            return queue_view.addOne.wasCalled;
        }, 1200, "calling Queue addOne again");
        runs(function(){
            expect(queue_view.addOne).toHaveBeenCalled();
        });
    });
    it("should show a list of errors", function(){
        runs(function(){
            expect($('#nav_monitor .errors .cell div').length).toBe(0);
            errQueueItem(2);
        });
        waitsFor(function(){
            return $('#nav_monitor .errors .cell div').length;
        }, 1200, "show list of errors");
        runs(function(){
            expect($('#nav_monitor .errors .cell div').length).not.toBe(0);
        });
    });
    it("should change length on queue item completion", function(){
        var len;
        runs(function(){
            len = queue_view.length();
            completeQueueItem(1);
        });
        waitsFor(function(){
            return queue_view.length() != len;
        }, 1200, "changing queue_view length");
        runs(function(){
            expect(len).not.toEqual(queue_view.length());
        });
    });
    it("should show an error log", function(){
        runs(function(){
            expect($('#error_popup .error').length).toBe(0);
            $('#nav_monitor .errors .cell div div .queue_item_log').trigger("click");
        });
        waitsFor(function(){
            return $('#error_popup .error').length;
        }, 1200, "showing error popup");
        runs(function(){
            expect($('#error_popup .error').length).not.toBe(0);
            $('#error_popup .error div div button').trigger("click");
        });
        waitsFor(function(){
            return !($('#error_popup .error').length);
        }, 1200, "closing error popup");
        runs(function(){
            expect($('#error_popup .error').length).toBe(0);
        });
    });
    it("should delete error items", function(){
        runs(function(){
            $('#nav_monitor .errors .cell div div .queue_remove').trigger("click");
        });
        waitsFor(function(){
            return !($('#nav_monitor .errors .cell div').length);
        }, 1200, "show list of errors");
        runs(function(){
            expect($('#nav_monitor .errors .cell div').length).toBe(0);
        });
    });
    xit("should remove any queue item", function(){
        runs(function(){
            queue_view.addOne.wasCalled = false;
            processQueueItem(3);
            waits(100);
            $('#top_monitor .cell div div .queue_remove').trigger("click");
            waits(100);
            expect(queue_view.addOne.wasCalled).toBeTruthy();
        });
        waitsFor(function(){
            return !($('#top_monitor .cell div div').length);
        }, 1200, "changing queue_view length");
        runs(function(){
            expect($('#top_monitor .cell div div').length).toBe(0);
        });
    });
});

xdescribe("resource usage", function(){
    it("should not be running", function(){
        var usage = window.requirejs("usage");
        expect(usage.running).toBeFalsy();
    });
    it("should start running", function(){
        runs(function(){
            processQueueItem(4);
        });
        waitsFor(function(){
            return $('#top_monitor .cell div div').length;
        }, 2000, "queue item 4 starting");
        runs(function(){
            expect(usage.running).toBeTruthy();
        });
    });
});

var modeling_view;
describe("modeling view", function(){
    it("should exist", function(){
        modeling_view = nav_view.modeling_view;
        expect(modeling_view).toBeDefined();
    });
    var targets = {
        ".modelplus":["modelplus"],
        ".samplesizeplus":["samplesizeplus"],
        ".samplesizecancel":["samplesizecancel"],
        "#blender":["blender"],
        ".mexport":["mexport"],
        ".mpredict":["mpredict"],
        ".mdl_sel":[
            "highlight",
            "mdl_sel",
        ],
        "#actions>div":[
            "mdl_sel_actions_close",
            "mdl_sel_actions_change"
        ],
        ".modelKickOff":[
            "modelKickOff",
            "runMore"
        ]
    };
    for(var target in targets){
        it("should have "+target, function(){
            waitsFor(function(){
                return $(target).length;
            }, "should exist in DOM", 1000);
            runs(function(){
                expect($(target).length).not.toBe(0);
            });
        });
        it("should be able to safely click "+target, function(){
            var opts = [];
            var reqs = [];
            var targetlen = 0;
            runs(function(){
                for(var f in targets[target]){
                    var func = targets[target][f];
                    if($.isArray(func)){
                        for(var g in func){
                            opts.unshift(spyOn(modeling_view, func[g]).andCallThrough());
                            
                        }
                    }else{
                        reqs.unshift(spyOn(modeling_view, func).andCallThrough());
                    }
                }
                modeling_view.delegateEvents(); // reset functions in View to call the spies
                targetlen = $(target).length;
                $(target).reverse().each(function(i){
                    $(this).trigger("mouseenter");
                    waits(20);
                    $(this).trigger("click");
                    waits(100);
                    $(this).trigger("mouseleave");
                    waits(30);
                });
            });
            runs(function(){
                for(var f in opts){
                    expect(opts[f]).toHaveBeenCalled();
                }
                for(f in reqs){
                    expect(reqs[f].callCount).toBe(targetlen);
                }
            });
        });
    }
});

/*
process.stdin.resume();
process.stdin.setEncoding('utf8');
 
process.stdin.on('data', function (chunk) {
 process.stdout.write('data: ' + chunk);
});*/
