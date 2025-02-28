"use strict"

(function() {
    // Create mesh holders.
    let sharedAttributes = {
        ar: true,
        "ar-modes": "webxr scene-viewer quick-look",
        loading: "lazy",
        reveal: "manual",
        // crossorigin: "anonymous",
        style: "height: 300px; width: 100%;",
        "camera-controls": true,
        "touch-action": "pan-y",
        "shadow-intensity": "1",
        exposure: "1"
    };

    let meshAttributes = {
      "toy": {
        src: "https://anonymous000edm.github.io/static/ply/ex.ply",
        poster: "https://anonymous000edm.github.io/static/ply/ex.png",
        "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
        caption: "example",
        shortCaption: "example",
      },
    };

    let meshRows = [
      ['toy', 'toy'],
      ['toy', 'toy'],
      ['toy', 'toy'],
      ['toy', 'toy'],
    ];

    // let meshAttributes = {
    //     chick: {
    //         src: "https://dreamfusion3d.github.io/assets/meshes2/44855521_sept18_hero16_047a_DSLR_photo_of_an_eggshell_broken_in_two_with_an_adorable_chick_standing_next_to_it_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44855521_sept18_hero16_047a_DSLR_photo_of_an_eggshell_broken_in_two_with_an_adorable_chick_standing_next_to_it_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a DSLR photo of an eggshell broken in two with an adorable chick standing next to it",
    //         shortCaption: "[...] eggshell broken in two with an adorable chick standing next to it"
    //     },
    //     pig: {
    //         src: "/assets/meshes2/44844973_sept18_hero14_076a_pig_wearing_a_backpack_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44844973_sept18_hero14_076a_pig_wearing_a_backpack_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a pig wearing a backpack",
    //         shortCaption: "a pig wearing a backpack"
    //     },
    //     frog: {
    //         src: "/assets/meshes2/sweaterfrog_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/sweaterfrog_1step.jpg",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a DSLR photo of a frog wearing a sweater",
    //         shortCaption: "[...] frog wearing a sweater",
    //     },
    //     lemur: {
    //         src: "/assets/meshes2/44853505_sept18_hero15_124a_lemur_taking_notes_in_a_journal_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44853505_sept18_hero15_124a_lemur_taking_notes_in_a_journal_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a lemur taking notes in a journal",
    //         shortCaption: "a lemur taking notes in a journal",
    //     },
    //     eagle: {
    //         src: "/assets/meshes2/44853505_sept18_hero15_145a_bald_eagle_carved_out_of_wood_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44853505_sept18_hero15_145a_bald_eagle_carved_out_of_wood_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a bald eagle carved out of wood",
    //         shortCaption: "a bald eagle carved out of wood",
    //     },
    //     crab: {
    //         src: "/assets/meshes2/44930695_sept18_hero18_103a_crab,_low_poly_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44930695_sept18_hero18_103a_crab,_low_poly_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/whipple_creek_regional_park_04_1k.hdr",
    //         caption: "a crab, low poly",
    //         shortCaption: "a crab, low poly",
    //     },
    //     ghost: {
    //         src: "/assets/meshes2/44934035_sept18_hero19_113a_DSLR_photo_of_a_ghost_eating_a_hamburger_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44934035_sept18_hero19_113a_DSLR_photo_of_a_ghost_eating_a_hamburger_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a DSLR photo of a ghost eating a hamburger",
    //         shortCaption: "[...] ghost eating a hamburger",
    //     },
    //     corgi: {
    //         src: "/assets/meshes2/44960400_sept18_hero20peter_117a_plush_toy_of_a_corgi_nurse_1step.glb",
    //         poster: "https://dreamfusion-cdn.ajayj.com/mesh_previews2/44960400_sept18_hero20peter_117a_plush_toy_of_a_corgi_nurse_1step.png",
    //         "environment-image": "https://modelviewer.dev/shared-assets/environments/spruit_sunrise_1k_HDR.hdr",
    //         caption: "a plush toy of a corgi nurse",
    //         shortCaption: "a plush toy of a corgi nurse",
    //     },
    // };

    // let meshRows = [
    //     ['frog', 'chick'],
    //     ['ghost', 'pig'],
    //     ['eagle', 'crab'],
    //     ['lemur', 'corgi'],
    // ];

    let container = document.getElementById("meshContainer");
    meshRows.forEach((meshIds) => {
        let row = document.createElement("DIV");
        row.classList = "row";

        meshIds.forEach((meshId) => {
            let col = document.createElement("DIV");
            col.classList = "col-md-6 col-sm-6 my-auto";
            
            // Model viewer.
            let model = document.createElement("model-viewer");
            for (const attr in sharedAttributes) {
                if (attr != "caption" && attr != "shortCaption")
                    model.setAttribute(attr, sharedAttributes[attr]);
            }
            for (const attrCustom in meshAttributes[meshId]) {
                if (attrCustom != "caption" && attrCustom != "shortCaption")
                    model.setAttribute(attrCustom, meshAttributes[meshId][attrCustom]);
            }
            model.id = 'mesh-' + meshId;

            // Controls.
            let controls = document.createElement("div");
            controls.className = "controls";
            let buttonLoad = document.createElement("button");
            buttonLoad.classList = "btn btn-primary loads-model";
            buttonLoad.setAttribute("data-controls", model.id);
            buttonLoad.appendChild(document.createTextNode("Load 3D model"));
            // let buttonToggle = document.createElement("button");
            // buttonToggle.classList = "btn btn-primary toggles-texture";
            // buttonToggle.setAttribute("data-controls", model.id);
            // buttonToggle.appendChild(document.createTextNode("Toggle texture"));
            controls.appendChild(buttonLoad);
            // controls.appendChild(buttonToggle);

            // Caption.
            let caption = document.createElement("p");
            caption.classList = "caption";
            caption.title = meshAttributes[meshId]["caption"] || "";
            caption.appendChild(document.createTextNode(meshAttributes[meshId]["shortCaption"] || caption.title));

            col.appendChild(model);
            col.appendChild(controls);
            col.appendChild(caption);
            row.appendChild(col);
        });

        container.appendChild(row);
    });

    // Toggle texture handlers.
    document.querySelectorAll('button.toggles-texture').forEach((button) => {
        button.addEventListener('click', () => {
            let model = document.getElementById(button.getAttribute("data-controls"));

            console.log(model.model);
            let material = model.model.materials[0];
            let metallic = material.pbrMetallicRoughness;
            let originalTexture = metallic.pbrMetallicRoughness.baseColorTexture;
            let originalBaseColor = metallic.pbrMetallicRoughness.baseColorFactor;
            console.log('model load', model.model, material, 'metallic', metallic, originalTexture);

            let textureButton = model.querySelector('.toggles-parent-texture');
            console.log('texture button', textureButton);
            // if (originalTexture && textureButton) {
            let textureOn = true;
            textureButton.onclick = () => {
                if (textureOn) {
                    // model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
                    // model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor([1., 1., 1., 1.]);
                    // textureOn = false;
                    // console.log('toggle texture off');
                } else {
                    // model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
                    // model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor(originalBaseColor);
                    // textureOn = true;
                    // console.log('toggle texture on');
                }
            };
        });
    });

    // Click to load handlers for 3D meshes.
    document.querySelectorAll('button.loads-model').forEach((button) => {
        button.setAttribute('data-action', 'load');
        button.addEventListener('click', () => {
            // button.classList = button.classList + " disappearing";
            // let model = button.parentElement.parentElement;
            let model = document.getElementById(button.getAttribute("data-controls"));

            if (button.getAttribute('data-action') == 'load') {
                model.dismissPoster();
                button.classList = "btn btn-disabled";
                button.innerHTML = "Hide 3D model";
                button.setAttribute('data-action', 'unload');
            } else {
                model.showPoster();
                button.classList = "btn btn-primary";
                button.innerHTML = "Load 3D model";
                button.setAttribute('data-action', 'load');
            };
        });
    });
    // document.querySelectorAll('button.toggles-parent-texture').forEach((button) => {
    //     let model = button.parentElement.parentElement;
    //     let originalTexture = model.materials[0].pbrMetallicRoughness.baseColorTexture;
    //     let textureOn = true;
    //     button.addEventListener('click', () => {
    //         if (textureOn) {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    //             textureOn = false;
    //         } else {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
    //             textureOn = true;
    //         }
    //     });
    // });
})();
