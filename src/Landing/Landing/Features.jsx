import React from 'react'
import { IoIosFitness} from "react-icons/io";
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import { AiOutlineSafetyCertificate } from "react-icons/ai";
import nhf from './no_hidden_fees.png'

const Features = () => {


    const bag2 = "https://assets.website-files.com/62cc07ca0720bd63152e1799/62cd16b4a5613c06cf9a0ff4_line-bg.svg";

    const data = [
        {
            logo: <IoIosFitness size={40}/>,
            heading: "Interactive Fitness Plans",
            content: "Engaging users with interactive and fun fitness plans that make achieving wellness goals a fun and enjoyable experience."
        },
        {
            logo: <MonitorHeartIcon fontSize='large'  />,
            heading: "Real-time Health Tracking",
            content: "Providing users with real-time health updates and tracking tools, empowering them to make informed choices for their well-being."
        },
        {
            logo: nhf,
            heading: "Rewarding Achievements",
            content: "Creating a sense of accomplishment and motivation through challenges, rewards system to achieve fitness milestones."
        },
        {
            logo: <AiOutlineSafetyCertificate size={40}/>,
            heading: "Certificates and Badges",
            content: "Offering tangible recognition of their fitness achievements with certificates and badges upon reaching health and wellness milestones."
        },
    ]
  return (
    <div style={{backgroundImage: `url(${bag2})`, backgroundSize:'cover'}} className='h-auto flex flex-col items-center justify-center py-5'>
        
        <h1 className='text-5xl  font-bold text-center py-5'>Fun and Exciting Features</h1>
        <p className='md:text-xl text-lg italic font-semibold text-gray-500 text-center my-5 px-10'>Achieve your fitness goals with ease. Through our fitness app, <br/> you can track and manage your health effortlessly.</p>

        <div className='grid md:grid-cols-4 grid-cols-1 place-items-center items-center justify-center m-10 '>
            
            {data.map((data) =>(

            <div className='mx-5 p-6 bg-white rounded-2xl transition duration-700 hover:-translate-y-3 hover:bg-[#1976D2] hover:text-white group shadow-2xl'>

                {data.logo === nhf ? <img src={data.logo} className='h-12'/> : <span>{data.logo}</span>}
                <h1 className='text-2xl font-bold my-5'>{data.heading}</h1>
                <p className=' text-lg font-bold italic text-gray-300 transition duration-700 group-hover:text-white'>{data.content}</p>

            </div>

            ))}

        </div>


    </div>
  )
}

export default Features