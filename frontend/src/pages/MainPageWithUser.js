import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import '../styles/MainPage.css';
import DashBoardPage from './DashBoardPage';

export default function MainPageWithUser({ user, setUser }) {
    const location = useLocation();
    const navigate = useNavigate();
    
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const user_id = params.get("user_id");
        const name = params.get("nickname");
        const profile_image = params.get("profile_image");
        const provider = params.get("provider");

        if (name) {
            setUser({ user_id, name, profile_image, provider });
        }
    }, [location, setUser]);

    useEffect(() => {
        if (user) {
            navigate("/");
        }
    }, [user, navigate]);

    return (
        <div>
            <DashBoardPage user={user}/>
        </div>
    );
}